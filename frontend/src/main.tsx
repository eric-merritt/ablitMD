import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";
import { ChartDemo } from "./components/organisms/ChartDemo.tsx";

const Root = window.location.hash === '#demo' ? ChartDemo : App

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
);
