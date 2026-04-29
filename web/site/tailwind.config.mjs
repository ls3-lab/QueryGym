import sharedPreset from "@qg/shared/tailwind.preset";

/** @type {import('tailwindcss').Config} */
export default {
  presets: [sharedPreset],
  content: [
    "./src/**/*.{astro,html,js,jsx,ts,tsx,mdx}",
    "../../web/shared/components/**/*.{astro,js,ts}",
  ],
};
