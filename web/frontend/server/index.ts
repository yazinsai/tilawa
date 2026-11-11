import { serve } from "@hono/node-server";
import { createApp } from "./app.js";

const { app } = createApp();
const port = parseInt(process.env.PORT || "5000");
console.log(`Server running on port ${port}`);
serve({ fetch: app.fetch, port });
