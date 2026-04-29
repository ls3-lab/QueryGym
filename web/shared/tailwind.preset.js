/** Tailwind preset extended by every QueryGym site. */
export default {
  darkMode: ["selector", '[data-theme="dark"]'],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      colors: {
        qg: {
          "grad-start": "var(--qg-grad-start)",
          "grad-end": "var(--qg-grad-end)",
          accent: "var(--qg-accent)",
          bg: "var(--qg-bg)",
          "bg-soft": "var(--qg-bg-soft)",
          fg: "var(--qg-fg)",
          "fg-muted": "var(--qg-fg-muted)",
          border: "var(--qg-border)",
        },
      },
      backgroundImage: {
        "qg-gradient":
          "linear-gradient(135deg, var(--qg-grad-start) 0%, var(--qg-grad-end) 100%)",
      },
    },
  },
  plugins: [],
};
