import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";
import solid from "@astrojs/solid-js";
import sitemap from "@astrojs/sitemap";

export default defineConfig({
  site: "https://leaderboard.querygym.com",
  output: "static",
  integrations: [
    tailwind({ applyBaseStyles: false }),
    solid(),
    sitemap({
      changefreq: "weekly",
      priority: 0.5,
      lastmod: new Date(),
      serialize(item) {
        // Home page is the canonical entry point.
        if (item.url === "https://leaderboard.querygym.com/") {
          return { ...item, priority: 1.0, changefreq: "weekly" };
        }
        // Per-run detail pages are mostly internal-link targets — give them
        // lower priority so search engines focus on the dataset/method/model
        // index pages.
        if (item.url.startsWith("https://leaderboard.querygym.com/runs/")) {
          return { ...item, priority: 0.3, changefreq: "monthly" };
        }
        return item;
      },
    }),
  ],
  vite: {
    ssr: {
      // Needed because we read CSV/YAML at build time from outside the project root.
      noExternal: ["@qg/shared"],
    },
  },
});
