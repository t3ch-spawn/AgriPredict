/* eslint-disable */

import React, { useEffect, useState } from "react";
import { useCurrTab } from "./CurrTabContext";
import Card from "./Card";
import { formatCurrency } from "./Overview";
import { useFetch } from "./useFetch";
import Loader from "./Loader";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Label,
} from "recharts";

export default function Comparison() {
  const { currCommodity, currState } = useCurrTab();
  const years = ["2026", "2027", "2028", "2029", "2030"];
  const [currYear, setCurrYear] = useState("2026");

  const { data: compareData, loading } = useFetch(
    `/compare/${currYear}/${currCommodity}`
  );

  const [insightData, setInsightData] = useState();

  const insights = [
    { insight: "Cheapest State", state: "Kano", value: 45000 },
    {
      insight: "Most Expensive State",
      state: "Lagos",
      value: 62000,
    },
    {
      insight: "Best Buying Period",
      state: "",
      value: "October",
    },
    { insight: "Yearly Average", state: "", value: 51500 },
  ];

  function toggleBubble(e) {
    const allBubbles = document.querySelectorAll(".select-bubble");

    allBubbles.forEach((bubble) => {
      bubble.classList.remove("active");

      if (bubble == e.currentTarget) {
        bubble.classList.add("active");
      }
    });
  }

  useEffect(() => {
    if (compareData) {
      console.log(compareData);
      setInsightData(compareData.analytics);
    }
  }, [compareData]);

  useEffect(() => {
    setInsightData();
  }, [currCommodity]);

  return (
    <div>
      <h1 className="heading">State Comparison for {currCommodity}</h1>

      <p className="text-[18px] mt-[20px]">
        Compare food commodity prices across different States
      </p>

      <Card className="mt-[30px] p-[20px] ">
        <p className="font-medium">Select Forecast Year</p>
        <div className="flex gap-[20px] mt-[10px]">
          {years.map((year, idx) => {
            return (
              <button
                onClick={(e) => {
                  toggleBubble(e);
                  setCurrYear(e.currentTarget.textContent);
                  setInsightData();
                }}
                className={`bg-greenSide bg-opacity-30 px-[10px] py-[5px] rounded-sm w-fit hover:bg-greenSide hover:text-black hover:font-medium duration-200 select-bubble ${
                  idx == 0 && "active"
                }`}
              >
                {year}
              </button>
            );
          })}
        </div>
      </Card>

      {loading || !insightData ? (
        <div className="flex justify-center items-center mt-[40px]">
          <Loader size={100} />
        </div>
      ) : (
        <>
          {" "}
          {/* Container for cards showing insights */}
          <div className="flex flex-wrap justify-start items-center gap-[30px] mt-[40px]">
            <InsightCard
              insight="Most Expensive State"
              state={insightData?.most_expensive_state.state}
              value={insightData?.most_expensive_state?.average_price}
            />
            <InsightCard
              insight="Cheapest State"
              state={insightData?.cheapest_state.state}
              value={insightData?.cheapest_state?.average_price}
            />
            <InsightCard
              insight="Best Buying Period"
              state={insightData?.best_month_to_buy.month_name}
              value={insightData?.best_month_to_buy?.average_price}
            />
          </div>
          <Card className="mt-[40px] p-[20px]">
            <p className="text-[18px] font-medium ">
              Price Trends Across States
            </p>

            <ForecastComparison graph_data={compareData.graph_data} />
          </Card>
        </>
      )}
    </div>
  );
}

function InsightCard({ insight, value, state }) {
  return (
    <Card className="flex flex-col justify-start items-start gap-[5px] px-[25px] py-[30px]">
      {/* insight */}
      <p className="text-[16px] font-medium">{insight}</p>

      {/* Value */}
      <p className="text-[28px] font-semibold">
        {state && `${state} - `}
        {formatCurrency(value)}
      </p>
    </Card>
  );
}

function ForecastComparison({ graph_data }) {
  // Flatten the data so all states share the same x-axis (month)
  const chartData = graph_data[0].data.map((_, i) => {
    const row = { month: graph_data[0].data[i].month };

    graph_data.forEach((item) => {
      row[item.state] = item.data[i].predicted_price;
    });

    return row;
  });

  function generateTicks(min, max, interval) {
    const ticks = [];
    let value = Math.floor(min / interval) * interval;

    while (value <= max) {
      ticks.push(value);
      value += interval;
    }

    return ticks;
  }

  const allPrices = graph_data.flatMap((s) =>
    s.data.map((p) => p.predicted_price)
  );

  const minPrice = Math.min(...allPrices);
  const maxPrice = Math.max(...allPrices);

  // 👉 choose your spacing here: 50, 100, etc.
  const customTicks = generateTicks(minPrice, maxPrice, 100);

  return (
    <LineChart
      width={800}
      height={400}
      data={chartData}
      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
    >
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="month">
        <Label value="Month" offset={-10} position="insideBottom" />
      </XAxis>
      <YAxis
        ticks={customTicks}
        tickFormatter={(value) =>
          value.toLocaleString("en-NG", {
            style: "currency",
            currency: "NGN",
            minimumFractionDigits: 0,
          })
        }
        interval={0} // Show all ticks
        tickCount={4} // Approx. number of ticks (controls spacing)
      >
        <Label
          value="Price (₦)"
          angle={-90}
          offset={-10}
          position="insideLeft"
          style={{ textAnchor: "middle" }}
        />
      </YAxis>
      <Tooltip />
      <Legend />

      {graph_data.map((item, index) => (
        <Line
          key={item.state}
          type="monotone"
          dataKey={item.state}
          stroke={["#8884d8", "#82ca9d", "#ff7300"][index % 3]} // cycle colors
          strokeWidth={2}
          dot={false}
        />
      ))}
    </LineChart>
  );
}
