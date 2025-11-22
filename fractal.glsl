#version 330

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

    // Modulate rotation or scale with audio if desired, 
    // or just let python drive the uniforms.
    // Here we add audio_level to the iteration color offset for a glow effect
    
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
