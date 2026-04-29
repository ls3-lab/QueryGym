import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";
import solid from "@astrojs/solid-js";

export default defineConfig({
  site: "https://leaderboard.querygym.com",
  output: "static",
  integrations: [tailwind({ applyBaseStyles: false }), solid()],
  vite: {
    ssr: {
      // Needed because we read CSV/YAML at build time from outside the project root.
      noExternal: ["@qg/shared"],
    },
  },
});
