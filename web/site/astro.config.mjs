import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";
import sitemap from "@astrojs/sitemap";

export default defineConfig({
  site: "https://querygym.com",
  output: "static",
  integrations: [
    tailwind({ applyBaseStyles: false }),
    sitemap({
      // Daily change frequency for content, weekly for the rest. Defaults are
      // conservative; bump priority for the home page so search engines treat
      // it as the canonical entry point.
      changefreq: "weekly",
      priority: 0.7,
      lastmod: new Date(),
      serialize(item) {
        if (item.url === "https://querygym.com/") {
          return { ...item, priority: 1.0, changefreq: "weekly" };
        }
        return item;
      },
    }),
  ],
  vite: {
    ssr: {
      noExternal: ["@qg/shared"],
    },
  },
});
