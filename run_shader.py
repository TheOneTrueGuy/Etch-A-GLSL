import moderngl
import moderngl_window as mglw
from moderngl_window.opengl.vao import VAO
import imgui
from PIL import Image
import json
import os
import numpy as np
import glob
import requests
import threading
import datetime
import re

# Optional Audio Import
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Audio features disabled (sounddevice not found).")

EXAMPLE_SHADER = """#version 330
uniform vec2 resolution;
uniform float time;
uniform float rot_speed;    // Control rotation speed
uniform float hue_base;     // Base hue for color
uniform float scale_factor; // Adjust scaling in the fractal
uniform vec3 param_c;       // Fractal parameter C (default: 3.0, 4.3, 1.4)
uniform vec3 param_d;       // Fractal parameter D (default: 3.7, 2.0, 2.0)
uniform float audio_level;  // Audio reactivity (0.0 to 1.0+)

out vec4 fragColor;

mat3 rotate3D(float angle, vec3 axis) {
    vec3 a = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float r = 1.0 - c;
    return mat3(
        a.x * a.x * r + c, a.y * a.x * r + a.z * s, a.z * a.x * r - a.y * s,
        a.x * a.y * r - a.z * s, a.y * a.y * r + c, a.z * a.y * r + a.x * s,
        a.x * a.z * r + a.y * s, a.y * a.z * r - a.x * s, a.z * a.z * r + c
    );
}

vec3 hsv(float h, float s, float v) {
    vec4 t = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(vec3(h) + t.xyz) * 6.0 - t.www);
    return v * mix(vec3(1.0), clamp(p - vec3(1.0), 0.0, 1.0), s);
}

void main() {
    vec2 r = resolution;
    float t = time;
    vec4 o = vec4(0.0);
    vec2 FC = gl_FragCoord.xy;

    for (float i = 0.0, g = 0.0, e = 0.0, s = 0.0; ++i < 44.0; ) {
        vec3 p = vec3((FC.xy - 0.5 * r) / r.y * 2.0, g - 0.5) * rotate3D(t * rot_speed, vec3(4.0, 4.0, cos(t * rot_speed)));
        s = 2.0;
        for (int j = 0; j++ < 49; p = param_c - abs(abs(p) * e - param_d))
            s *= e = max(1.0, scale_factor / dot(p, p));
        g += mod(length(p.zx), p.y) / s;
        s = log2(s) / g;
        o.rgb += hsv(hue_base, e - i * 0.07 + audio_level * 0.5, s / 1e4);
    }
    fragColor = o;
}
"""

SYSTEM_PROMPT = f"""You are an expert GLSL shader programmer and digital artist.
Your task is to write a CREATIVE and VISUALLY STUNNING fragment shader compatible with OpenGL 3.3.

STANDARD UNIFORMS (Use these to drive the animation and interactivity):
uniform vec2 resolution;       // Viewport resolution (pixels)
uniform float time;            // Time in seconds
uniform float rot_speed;       // Rotation speed factor (0.0 to 2.0) - use to control animation speed or rotation
uniform float hue_base;        // Base hue (0.0 to 1.0) - use for color cycling
uniform float scale_factor;    // Scale factor (1.0 to 20.0) - use for zooming or intensity
uniform vec3 param_c;          // Custom generic parameter (vec3) - map to something interesting!
uniform vec3 param_d;          // Custom generic parameter (vec3) - map to something interesting!
uniform float audio_level;     // Audio reactivity level (0.0 to 1.0+) - make it pulse to the beat!

OUTPUT:
out vec4 fragColor;

EXAMPLE OF VALID CODE STRUCTURE:
{EXAMPLE_SHADER}

INSTRUCTIONS:
1. Write valid GLSL 3.30 code.
2. Ensure the shader compiles.
3. Be creative! Raymarching, SDFs, fractals, plasma, fluids, reaction-diffusion, etc.
4. Use the provided uniforms to make it interactive.
5. RESPOND ONLY WITH THE CODE BLOCK inside markdown fences (```glsl ... ```).
"""

class FractalWindow(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Etch-A-GLSL: Procedural Fractal"
    window_size = (1280, 720)
    aspect_ratio = 16 / 9
    resizable = True

    def __init__(self, **kwargs):
        print("Starting __init__", flush=True)
        super().__init__(**kwargs)
        
        # Shader loading
        self.shader_files = sorted(glob.glob("*.glsl"))
        self.current_shader_idx = 0
        if "fractal.glsl" in self.shader_files:
            self.current_shader_idx = self.shader_files.index("fractal.glsl")
        
        self.shader_error = None
        
        # AI Generation State
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.user_prompt = "A cyberpunk city with neon rain"
        self.is_generating = False
        self.gen_status = ""
        self.new_shader_filename = None # To communicate back to main thread

        # Simple vertex shader for full-screen quad
        self.vert_shader = '''
        #version 330
        in vec2 in_position;
        out vec2 uv;
        void main() {
            uv = (in_position + 1.0) / 2.0;
            gl_Position = vec4(in_position, 0.0, 1.0);
        }
        '''

        self.prog = None
        if self.shader_files:
            self.load_shader(self.shader_files[self.current_shader_idx])

        # Full-screen quad geometry
        vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4')
        self.quad = VAO(name='quad')
        self.quad.buffer(vertices, '2f', ['in_position'])
        self.quad.index_buffer(np.array([0, 1, 2, 1, 3, 2], dtype='i4'))

        # Initial parameter values
        self.params = {
            'rot_speed': 0.5,
            'hue_base': 0.6,
            'scale_factor': 9.0,
            'param_c': [3.0, 4.3, 1.4],
            'param_d': [3.7, 2.0, 2.0]
        }
        
        self.mouse_interaction = False
        self.audio_enabled = False
        self.audio_level = 0.0
        self.audio_stream = None
        
        if AUDIO_AVAILABLE:
            try:
                self.init_audio()
            except Exception as e:
                print(f"Audio init failed: {e}")

        self.load_presets()
        self.preset_name = "New Preset"
        
        # Manual ImGui Integration Setup
        try:
            import imgui
            from moderngl_window.integrations.imgui import ModernglWindowRenderer
            imgui.create_context()
            self.imgui = ModernglWindowRenderer(self.wnd)
            print("Manual ImGui init successful", flush=True)
            
            # Force-wire Pyglet events if they aren't working
            if self.wnd.name == 'pyglet':
                print("Forcing Pyglet event wiring via _window.push_handlers", flush=True)
                if hasattr(self.wnd, '_window'):
                    self.wnd._window.push_handlers(self)
        except Exception as e:
            print(f"Manual ImGui/Event init failed: {e}", flush=True)

        print("Init complete", flush=True)

    # --- Pyglet Bridge Methods ---
    def on_mouse_motion(self, x, y, dx, dy):
        # Pyglet is Y-up, ImGui is Y-down. Flip Y.
        self.mouse_position_event(x, self.wnd.height - y, dx, -dy)
        
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        # Pyglet is Y-up, ImGui is Y-down. Flip Y.
        self.mouse_drag_event(x, self.wnd.height - y, dx, -dy)
        self.mouse_position_event(x, self.wnd.height - y, dx, -dy) 

    def on_mouse_press(self, x, y, button, modifiers):
        self.mouse_press_event(x, self.wnd.height - y, button)

    def on_mouse_release(self, x, y, button, modifiers):
        self.mouse_release_event(x, self.wnd.height - y, button)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.mouse_scroll_event(scroll_x, scroll_y)

    def on_key_press(self, symbol, modifiers):
        self.key_event(symbol, self.wnd.keys.ACTION_PRESS, modifiers)

    def on_key_release(self, symbol, modifiers):
        self.key_event(symbol, self.wnd.keys.ACTION_RELEASE, modifiers)
        
    def on_text(self, text):
        self.unicode_char_entered(text)

    def load_shader(self, shader_path):
        try:
            frag_source = open(shader_path).read()
            self.prog = self.ctx.program(vertex_shader=self.vert_shader, fragment_shader=frag_source)
            self.shader_error = None
            print(f"Loaded shader: {shader_path}")
        except Exception as e:
            self.shader_error = str(e)
            print(f"Error loading shader {shader_path}: {e}")

    def init_audio(self):
        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            # Simple volume meter
            volume_norm = np.linalg.norm(indata) * 10
            self.audio_level = float(volume_norm)

        self.audio_stream = sd.InputStream(callback=audio_callback, channels=1, blocksize=1024)

    def toggle_audio(self):
        if not AUDIO_AVAILABLE:
            return
        
        if self.audio_enabled:
            if self.audio_stream:
                self.audio_stream.stop()
            self.audio_enabled = False
        else:
            if self.audio_stream:
                self.audio_stream.start()
            self.audio_enabled = True

    def load_presets(self):
        self.presets = {}
        if os.path.exists('presets.json'):
            try:
                with open('presets.json', 'r') as f:
                    self.presets = json.load(f)
            except:
                print("Failed to load presets.")

    def save_presets(self):
        with open('presets.json', 'w') as f:
            json.dump(self.presets, f, indent=2)

    def apply_preset(self, name):
        if name in self.presets:
            p = self.presets[name]
            # Safely update params
            for k, v in p.items():
                if k in self.params:
                    self.params[k] = v

    def on_render(self, time: float, frame_time: float):
        # Reset standard GL state for the fractal render
        self.ctx.viewport = (0, 0, self.wnd.width, self.wnd.height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)
        
        self.ctx.clear(0.0, 0.0, 0.0)
        
        # Ensure ImGui frame is started via Integrator
        # The integrator's .render() will handle the draw data, 
        # but we still need new_frame(). The integrator might handle inputs.
        try:
             # We still need to start the frame, but let's see if the context is happier now
            import imgui
            
            # Force sync mouse buttons to prevent stuck state
            if self.imgui:
                self.imgui.io.mouse_down[0] = self.wnd.mouse_states.left
                self.imgui.io.mouse_down[1] = self.wnd.mouse_states.right
                self.imgui.io.mouse_down[2] = self.wnd.mouse_states.middle
            
            imgui.new_frame()
        except Exception as e:
            print(f"new_frame error: {e}", flush=True)

        # Check if generation thread finished
        if self.new_shader_filename:
            self.shader_files = sorted(glob.glob("*.glsl"))
            if self.new_shader_filename in self.shader_files:
                self.current_shader_idx = self.shader_files.index(self.new_shader_filename)
                self.load_shader(self.new_shader_filename)
            self.new_shader_filename = None

        if self.prog:
            # Update uniforms safely (check if they exist in current shader)
            try:
                if 'resolution' in self.prog:
                    self.prog['resolution'].value = (self.wnd.width, self.wnd.height)
                if 'time' in self.prog:
                    self.prog['time'].value = time
                
                # Only set custom params if they exist in the shader
                if 'rot_speed' in self.prog:
                    self.prog['rot_speed'].value = self.params['rot_speed']
                if 'hue_base' in self.prog:
                    self.prog['hue_base'].value = self.params['hue_base']
                if 'scale_factor' in self.prog:
                    self.prog['scale_factor'].value = self.params['scale_factor']
                if 'param_c' in self.prog:
                    self.prog['param_c'].value = tuple(self.params['param_c'])
                if 'param_d' in self.prog:
                    self.prog['param_d'].value = tuple(self.params['param_d'])
                if 'audio_level' in self.prog:
                    self.prog['audio_level'].value = self.audio_level if self.audio_enabled else 0.0

                self.quad.render(self.prog, mode=moderngl.TRIANGLES)
            except Exception as e:
                print(f"Render error: {e}")

        # Draw the UI
        try:
            self.imgui_gui()
        except Exception as e:
            print(f"GUI Definition error: {e}", flush=True)

        # Render UI
        try:
            imgui.render()
            if self.imgui:
                self.imgui.render(imgui.get_draw_data())
        except Exception as e:
            print(f"UI Render error: {e}", flush=True)

    def generate_shader_thread(self, prompt, api_key):
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/etch-a-glsl", # Required by OpenRouter
                "X-Title": "Etch-A-GLSL"
            }
            
            data = {
                "model": "anthropic/claude-3-haiku", # Cost-effective and good at code
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Design a new GLSL shader with this description: {prompt}"}
                ]
            }
            
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                
                # Extract code block
                match = re.search(r'```(?:glsl)?(.*?)```', content, re.DOTALL)
                if match:
                    glsl_code = match.group(1).strip()
                    
                    # Save to file
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"ai_gen_{timestamp}.glsl"
                    with open(filename, 'w') as f:
                        f.write(glsl_code)
                    
                    self.gen_status = f"Success! Saved {filename}"
                    self.new_shader_filename = filename # Signal main thread
                else:
                    self.gen_status = "Error: No code block found in response."
            else:
                self.gen_status = f"API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            self.gen_status = f"Exception: {str(e)}"
        finally:
            self.is_generating = False

    def imgui_gui(self):
        try:
            imgui.begin("Etch-A-GLSL Controls")
            
            # AI Generator
            if imgui.collapsing_header("AI Generator", visible=True)[0]:
                _, self.api_key = imgui.input_text("OpenRouter API Key", self.api_key, 256, flags=imgui.INPUT_TEXT_PASSWORD)
                
                imgui.text("Describe your shader:")
                _, self.user_prompt = imgui.input_text_multiline("##prompt", self.user_prompt, 200)
                
                if self.is_generating:
                    imgui.text_colored("Generating... Please wait...", 1.0, 1.0, 0.0)
                else:
                    if imgui.button("Generate Shader"):
                        if not self.api_key:
                            self.gen_status = "Error: API Key required!"
                        else:
                            self.is_generating = True
                            self.gen_status = "Starting..."
                            threading.Thread(target=self.generate_shader_thread, args=(self.user_prompt, self.api_key), daemon=True).start()
                
                if self.gen_status:
                    imgui.text_wrapped(self.gen_status)

            # Shader Selection
            if imgui.collapsing_header("Shader Selection", visible=True)[0]:
                changed, self.current_shader_idx = imgui.combo(
                    "Shader", self.current_shader_idx, self.shader_files
                )
                if changed:
                    self.load_shader(self.shader_files[self.current_shader_idx])
                
                if imgui.button("Refresh File List"):
                    self.shader_files = sorted(glob.glob("*.glsl"))
                    if self.current_shader_idx >= len(self.shader_files):
                        self.current_shader_idx = 0
                
                if self.shader_error:
                    imgui.text_colored(self.shader_error, 1.0, 0.0, 0.0)

            if imgui.collapsing_header("General Params", visible=True)[0]:
                _, self.params['rot_speed'] = imgui.slider_float("Rotation Speed", self.params['rot_speed'], 0.0, 2.0)
                _, self.params['hue_base'] = imgui.slider_float("Base Hue", self.params['hue_base'], 0.0, 1.0)
                _, self.params['scale_factor'] = imgui.slider_float("Scale Factor", self.params['scale_factor'], 1.0, 20.0)

            if imgui.collapsing_header("Fractal DNA", visible=True)[0]:
                changed_c, new_c = imgui.slider_float3("Param C", *self.params['param_c'], min_value=0.0, max_value=10.0)
                if changed_c: self.params['param_c'] = list(new_c)
                
                changed_d, new_d = imgui.slider_float3("Param D", *self.params['param_d'], min_value=0.0, max_value=10.0)
                if changed_d: self.params['param_d'] = list(new_d)

            if imgui.collapsing_header("Interaction", visible=True)[0]:
                _, self.mouse_interaction = imgui.checkbox("Mouse Interaction (Drag)", self.mouse_interaction)
                
                if AUDIO_AVAILABLE:
                    if imgui.button("Disable Audio" if self.audio_enabled else "Enable Audio"):
                        self.toggle_audio()
                    imgui.same_line()
                    imgui.text(f"Level: {self.audio_level:.2f}")
                else:
                    imgui.text("Audio library not found.")

                if imgui.button("Take Screenshot (S)"):
                    self.take_screenshot()

            if imgui.collapsing_header("Presets", visible=True)[0]:
                _, self.preset_name = imgui.input_text("Name", self.preset_name, 64)
                if imgui.button("Save Preset"):
                    self.presets[self.preset_name] = self.params.copy()
                    self.save_presets()
                
                imgui.separator()
                for name in list(self.presets.keys()):
                    if imgui.button(f"Load: {name}"):
                        self.apply_preset(name)
                    imgui.same_line()
                    if imgui.button(f"X##{name}"):
                        del self.presets[name]
                        self.save_presets()

        finally:
            imgui.end()

    def key_event(self, key, action, modifiers):
        if self.imgui:
            self.imgui.key_event(key, action, modifiers)
        super().key_event(key, action, modifiers)
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.S:
                self.take_screenshot()

    def mouse_position_event(self, x, y, dx, dy):
        # print(f"Mouse pos: {x}, {y}", flush=True)
        if self.imgui:
            self.imgui.mouse_position_event(x, y, dx, dy)

    def mouse_drag_event(self, x, y, dx, dy):
        if self.imgui:
            self.imgui.mouse_drag_event(x, y, dx, dy)
            
        if self.mouse_interaction:
            # Only rotate if ImGui isn't capturing the mouse
            if not imgui.get_io().want_capture_mouse:
                self.params['rot_speed'] += dx * 0.01
                self.params['scale_factor'] += dy * 0.1

    def mouse_scroll_event(self, x_offset, y_offset):
        # Manually handle scroll to avoid attribute error in moderngl-window integration
        # self.imgui.mouse_scroll_event(x_offset, y_offset)
        if self.imgui:
            self.imgui.io.mouse_wheel = y_offset

    def mouse_press_event(self, x, y, button):
        if self.imgui:
            self.imgui.mouse_press_event(x, y, button)

    def mouse_release_event(self, x: int, y: int, button: int):
        if self.imgui:
            self.imgui.mouse_release_event(x, y, button)

    def unicode_char_entered(self, char):
        if self.imgui:
            self.imgui.unicode_char_entered(char)

    def take_screenshot(self):
        import datetime
        filename = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        image = Image.frombytes('RGB', self.wnd.size, self.ctx.fbo.read(components=3))
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.save(filename)
        print(f"Saved {filename}")

    def close(self):
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
        super().close()

if __name__ == '__main__':
    mglw.run_window_config(FractalWindow)
