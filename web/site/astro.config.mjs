import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";

export default defineConfig({
  site: "https://querygym.com",
  output: "static",
  integrations: [tailwind({ applyBaseStyles: false })],
  vite: {
    ssr: {
      noExternal: ["@qg/shared"],
    },
  },
});
