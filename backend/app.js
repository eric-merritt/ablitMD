import express from "express";
import mongoose from "mongoose";
import modelsRouter from "./routes/api/model.js";
import promptsRouter from "./routes/api/prompts.js";
import runsRouter from "./routes/api/runs.js";
import inferenceRouter from "./routes/api/inference.js";
import dotenv from "dotenv";

dotenv.config();

const MONGO_URI = process.env.MONGO_URI;
if (!MONGO_URI) throw new Error("MONGO_URI is not set");

mongoose
  .connect(MONGO_URI, { serverSelectionTimeoutMS: 5000 })
  .then(() => console.log("MongoDB connected"))
  .catch((err) => {
    if (process.env.NODE_ENV !== "test") {
      console.error("MongoDB connection error:", err);
    }
  });

const app = express();

app.use(express.json({ limit: '256mb' }));

app.use("/api/models", modelsRouter);
app.use("/api/prompts", promptsRouter);
app.use("/api/runs", runsRouter);
app.use("/api/inference", inferenceRouter);

app.use((err, req, res, next) => {
  console.error(err);
  res.status(500).json({ error: err.message || "Internal server error" });
});

export default app;
