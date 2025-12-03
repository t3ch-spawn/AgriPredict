import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.jsx";
import { CurrTabProvider } from "./components/CurrTabContext";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <CurrTabProvider>
      <App />
    </CurrTabProvider>
  </StrictMode>
);
