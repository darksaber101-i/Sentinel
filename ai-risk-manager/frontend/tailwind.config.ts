import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Brand palette: warm neutrals + amber, tuned for a light, professional surface.
        bg:      "#f5f6f8",
        surface: "#ffffff",
        card:    "#ffffff",
        border:  "#e3e6ec",
        // DEFAULT is contrast-safe on white (text/icons/links). `solid` is the
        // brighter shade used only for filled buttons/pills paired with black text.
        amber: { DEFAULT: "#b45309", solid: "#f59e0b", dark: "#d97706", light: "#fef3c7" },
        text:  { primary: "#0f172a", secondary: "#475569", muted: "#94a3b8" },
        risk: {
          low:      "#16a34a",
          medium:   "#b45309",
          high:     "#ea580c",
          critical: "#dc2626",
        },
        money: { DEFAULT: "#047857", dark: "#065f46" },
      },
      fontFamily: { sans: ["Inter", "system-ui", "sans-serif"] },
      borderRadius: { xl: "12px", "2xl": "16px" },
    },
  },
  plugins: [],
};
export default config;
