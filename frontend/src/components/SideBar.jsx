import React from "react";
import { FaRegChartBar } from "react-icons/fa";
import { MdOutlineTrendingUp } from "react-icons/md";
import { MdPieChart } from "react-icons/md";
import { MdOutlineCompareArrows } from "react-icons/md";
import SelectBox from "./SelectBox";
import { useCurrTab } from "./CurrTabContext";

export default function SideBar() {
  const tabs = [
    { name: "Overview", icon: <FaRegChartBar color="#2D5016" /> },
    { name: "Forecast", icon: <MdOutlineTrendingUp color="#2D5016" /> },
    { name: "Analysis", icon: <MdPieChart color="#2D5016" /> },
    {
      name: "State Comparison",
      icon: <MdOutlineCompareArrows color="#2D5016" />,
    },
  ];

  const weights = {
    "Rice (local)": "2.7KG",
    "Beans (red)": "2.5KG",
    Yam: "2.5KG",
    Oranges: "400G",
  };

  const stateList = [
    { name: "Adamawa", value: "Adamawa" },
    { name: "Borno", value: "Borno" },
    { name: "Yobe", value: "Yobe" },
  ];

  const commodityList = [
    { name: "Rice (local)", value: "Rice (local)" },
    { name: "Beans (red)", value: "Beans (red)" },
    { name: "Yam", value: "Yam" },
    { name: "Oranges", value: "Oranges" },
  ];

  const {
    setCurrTab,
    currCommodity,
    currState,
    setCurrCommodity,
    setCurrState,
    currTab,
  } = useCurrTab();

  function toggleBubble(e) {
    const allBubbles = document.querySelectorAll(".sidebar-bubble");

    allBubbles.forEach((bubble) => {
      bubble.classList.remove("active");

      if (bubble == e.currentTarget) {
        bubble.classList.add("active");
      }
    });
  }

  return (
    <div className="h-[100vh] sticky px-[10px] pt-[20px] top-0 min-w-[250px] bg-greenSide">
      {/* Heading */}
      <h2 className="text-[28px] font-bold tracking-tight">AgriPredict NG</h2>

      {/* Container for list of tabs */}
      <div className="mt-[50px] flex flex-col gap-[10px]">
        {tabs.map((tab, idx) => {
          return (
            <div
              onClick={(e) => {
                setCurrTab(tab.name.toLowerCase());
                toggleBubble(e);
              }}
              className={`hover:bg-greenCard rounded-lg h-[40px] w-fit flex gap-[10px] justify-center items-center p-[5px] px-[10px] duration-300 cursor-pointer sidebar-bubble ${
                idx == 0 && "active"
              }`}
            >
              <div className="scale-[1.2]">{tab.icon}</div>
              <p className="font-medium">{tab.name}</p>
            </div>
          );
        })}
      </div>

      {/* Container for select boxes */}
      <div className="mt-[50px] flex flex-col">
        <p className="text-sm font-medium mb-[20px]">
          Select State and Commodity
        </p>

        {currTab != "state comparison" ? (
          <>
            {" "}
            <p className="text-[18px] mt-[20px] mb-[5px] font-medium">State</p>
            <SelectBox
              valueChange={(value) => {
                setCurrState(value);
              }}
              theme="State"
              defaultValue={currState}
              selectList={stateList}
            />
          </>
        ) : (
          ""
        )}

        <p className="text-[18px] mt-[20px] mb-[5px] font-medium">Commodity</p>
        <SelectBox
          valueChange={(value) => {
            setCurrCommodity(value);
          }}
          defaultValue={currCommodity}
          selectList={commodityList}
          theme="Commodity"
        />

        <p className="text-[18px] mt-[20px] mb-[5px] font-medium">Unit</p>
        <p className="text-[18px]  mb-[5px] font-medium">
          Per {weights[currCommodity]}
        </p>
      </div>
    </div>
  );
}
