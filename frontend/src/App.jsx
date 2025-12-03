import Forecast from "./components/Forecast";
import Overview from "./components/Overview";
import SideBar from "./components/SideBar";
import { useCurrTab } from "./components/CurrTabContext";
import Analysis from "./components/Analysis";
import Comparison from "./components/Comparison";
import { Toaster } from "sonner";

function App() {
  const { currTab } = useCurrTab();
  return (
    <main className="flex justify-start items-start w-full text-greenText">
      <SideBar />
      <Toaster />
      <div className="w-full px-[40px] pt-[30px] bg-greenBg min-h-[100vh]">
        {currTab == "overview" && <Overview />}
        {currTab == "forecast" && <Forecast />}
        {currTab == "analysis" && <Analysis />}
        {currTab == "state comparison" && <Comparison />}
      </div>
    </main>
  );
}

export default App;
