import express from "express";
import cors from "cors";
import testUserRoutes from "./routes/testUser.route.js";

const app = express();

app.use(cors());
app.use(express.json());

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    service: "backend",
  });
});

app.use("/api/test", testUserRoutes);

export default app;
