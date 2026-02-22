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
        bg: {
          DEFAULT: "#0f1117",
          surface: "#1a1d29",
          hover: "#22253a",
        },
        border: {
          DEFAULT: "#2a2d3a",
        },
        text: {
          primary: "#e6e8ed",
          secondary: "#8b8fa3",
        },
        green: {
          DEFAULT: "#00c853",
        },
        red: {
          DEFAULT: "#ff1744",
        },
        blue: {
          DEFAULT: "#2196f3",
        },
        warning: {
          DEFAULT: "#ffa000",
        },
      },
    },
  },
  plugins: [],
};
export default config;
