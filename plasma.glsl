#version 330

uniform vec2 resolution;
uniform float time;
uniform float scale_factor;
uniform float hue_base;

out vec4 fragColor;

void main() {
    vec2 uv = gl_FragCoord.xy / resolution;
    float t = time * 0.5;
    
    float v1 = sin(uv.x * 10.0 + t);
    float v2 = sin(uv.y * 10.0 + t);
    float v3 = sin((uv.x + uv.y) * 10.0 + t);
    float v4 = sin(length(uv - 0.5) * 10.0 + t);
    
    float v = v1 + v2 + v3 + v4;
    
    v = v * (scale_factor / 10.0); // Use scale slider to control intensity
    
    float r = sin(v + hue_base * 6.0);
    float g = sin(v + hue_base * 6.0 + 2.0);
    float b = sin(v + hue_base * 6.0 + 4.0);
    
    fragColor = vec4(r, g, b, 1.0) * 0.5 + 0.5;
}