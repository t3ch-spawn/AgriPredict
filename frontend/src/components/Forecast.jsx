import React, { useEffect, useState } from "react";
import Card from "./Card";
import Loader from "./Loader";
import { useCurrTab } from "./CurrTabContext";
import { FaPlusMinus } from "react-icons/fa6";
import { useFetch } from "./useFetch";
import { formatCurrency } from "./Overview";
import { toast } from "sonner";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function Forecast() {
  const years = ["2026", "2027", "2028", "2029", "2030"];
  const months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sept",
    "Oct",
    "Nov",
    "Dec",
  ];

  const [currYear, setCurrYear] = useState("");
  const [currMonth, setCurrMonth] = useState("");

  const [predictMonth, setPredictMonth] = useState(true);
  const { currState, currCommodity } = useCurrTab();

  const [fetchTrigger, setFetchTrigger] = useState(false);

  const { data: forecastData, loading: monthLoading } = useFetch(
    `/forecast_month/${currCommodity}/${currState}/${Number(currYear)}/${
      currMonth.num
    }`,
    fetchTrigger && predictMonth
  );
  const { data: forecastDataYear, loading: yearLoading } = useFetch(
    `/forecast_year/${currCommodity}/${currState}/${Number(currYear)}`,
    fetchTrigger && !predictMonth
  );
  const { data: analyticsData } = useFetch(
    `/analytics/${currState}/${currCommodity}`,
    fetchTrigger
  );

  useEffect(() => {
    if (forecastData) {
      console.log(forecastData);
    }
  }, [forecastData, analyticsData]);
  useEffect(() => {
    if (forecastDataYear) {
      console.log(forecastDataYear);
    }
  }, [forecastDataYear, analyticsData]);

  function toggleBubble(e) {
    const allBubbles = document.querySelectorAll(".select-bubble");

    allBubbles.forEach((bubble) => {
      bubble.classList.remove("active");

      if (bubble == e.currentTarget) {
        bubble.classList.add("active");
      }
    });
    setFetchTrigger(false);
  }

  function toggleBubble2(e) {
    const allBubbles = document.querySelectorAll(".select-bubble-2");

    allBubbles.forEach((bubble) => {
      bubble.classList.remove("active");

      if (bubble == e.currentTarget) {
        bubble.classList.add("active");
      }
    });
    setFetchTrigger(false);
  }

  function togglePredictType(e) {
    const allBubbles = document.querySelectorAll(".predict-type");

    allBubbles.forEach((bubble) => {
      bubble.classList.remove("active");

      if (bubble == e.currentTarget) {
        bubble.classList.add("active");
      }
    });
    setFetchTrigger(false);
  }

  return (
    <div className="flex flex-col pb-[100px]">
      <h1 className="heading">
        Price Forecast for {currCommodity} in {currState}
      </h1>

      {/* Tabs to select what manner of prediction */}
      <div className="flex items-center justify-start gap-[30px] mt-[10px]">
        <button
          onClick={(e) => {
            togglePredictType(e);
            setPredictMonth(true);
          }}
          className="border-greenText hover:scale-[1.05] border rounded-sm px-[10px] py-[4px] predict-type active"
        >
          Predict for a month in a future year{" "}
        </button>
        <button
          onClick={(e) => {
            togglePredictType(e);
            setPredictMonth(false);
          }}
          className="border-greenText hover:scale-[1.05] border rounded-sm px-[10px] py-[4px] predict-type"
        >
          Predict for a whole future year{" "}
        </button>
      </div>

      {predictMonth ? (
        <>
          {/* Select Forecast Year */}
          <Card className="mt-[30px] p-[20px] ">
            <p className="font-medium">Select Forecast Year</p>
            <div className="flex gap-[20px] mt-[10px]">
              {years.map((year) => {
                return (
                  <button
                    onClick={(e) => {
                      toggleBubble(e);
                      setCurrYear(e.currentTarget.textContent);
                    }}
                    className={`bg-greenSide bg-opacity-30 px-[10px] py-[5px] rounded-sm w-fit hover:bg-greenSide hover:text-black hover:font-medium duration-200 select-bubble `}
                  >
                    {year}
                  </button>
                );
              })}
            </div>
          </Card>

          {/* Select Forecast Month */}
          <Card className="mt-[20px] p-[20px] ">
            <p className="font-medium">Select Forecast Month</p>
            <div className="flex gap-[20px] mt-[10px]">
              {months.map((month) => {
                return (
                  <button
                    onClick={(e) => {
                      toggleBubble2(e);
                      setCurrMonth({
                        month: e.currentTarget.textContent,
                        num: months.indexOf(e.currentTarget.textContent) + 1,
                      });
                    }}
                    className={`bg-greenSide bg-opacity-30 px-[10px] py-[5px] rounded-sm w-fit hover:bg-greenSide hover:text-black hover:font-medium duration-200 select-bubble-2`}
                  >
                    {month}
                  </button>
                );
              })}
            </div>
          </Card>
        </>
      ) : (
        <>
          {/* Select Forecast Year */}
          <Card className="mt-[30px] p-[20px] ">
            <p className="font-medium">Select Forecast Year</p>
            <div className="flex gap-[20px] mt-[10px]">
              {years.map((year) => {
                return (
                  <button
                    onClick={(e) => {
                      toggleBubble(e);
                      setCurrYear(e.currentTarget.textContent);
                    }}
                    className={`bg-greenSide bg-opacity-30 px-[10px] py-[5px] rounded-sm w-fit hover:bg-greenSide hover:text-black hover:font-medium duration-200 select-bubble `}
                  >
                    {year}
                  </button>
                );
              })}
            </div>
          </Card>
        </>
      )}

      {/* Button to Generate the Forecast, i.e call the api */}
      <button
        onClick={() => {
          if (predictMonth) {
            if (currYear == "" || currMonth == "") {
              toast.error("Please select year and/or month");
            } else {
              setFetchTrigger(true);
            }
          } else if (!predictMonth) {
            if (currYear == "") {
              toast.error("Please select year");
            } else {
              setFetchTrigger(true);
            }
          }
        }}
        className="ml-auto w-fit bg-greenText text-white px-[12px] hover:scale-[1.05] duration-300 py-[8px] rounded-sm mt-[40px]"
      >
        Generate Forecast
      </button>

      {/* Forecast Chart */}

      {!monthLoading && predictMonth && fetchTrigger ? (
        <>
          <Card className="text-[36px] font-medium mt-[40px] p-[20px]">
            <span className="flex justify-start gap-0">
              <span>
                {" "}
                The forecast of {currCommodity} for the month of{" "}
                {currMonth.month} in {currYear} is{" "}
                {formatCurrency(forecastData?.predicted_price, 2)}
              </span>
              <span className="flex items-center ml-[5px]">
                (<FaPlusMinus className="text-[30px]" />{" "}
                {formatCurrency(analyticsData?.mae, 2)})
              </span>
            </span>
          </Card>
        </>
      ) : (
        ""
      )}

      {!yearLoading && !predictMonth && fetchTrigger ? (
        <Card className="mt-[30px]">
          <SingleLineChart data={forecastDataYear} />
        </Card>
      ) : (
        ""
      )}
    </div>
  );
}

export function SingleLineChart({ data }) {
  // Convert price strings to numbers
  const chartData = data.map((d) => ({
    ...d,
    price: Number(d.price),
  }));

  return (
    <ResponsiveContainer width="100%" height={350}>
      <LineChart
        data={chartData}
        margin={{ top: 20, right: 30, left: 20, bottom: 10 }}
      >
        <CartesianGrid strokeDasharray="3 3" />

        <XAxis dataKey="date" />

        <YAxis
          tickFormatter={(v) =>
            v.toLocaleString("en-NG", {
              style: "currency",
              currency: "NGN",
              minimumFractionDigits: 0,
            })
          }
        />

        <Tooltip
          formatter={(value) =>
            Number(value).toLocaleString("en-NG", {
              style: "currency",
              currency: "NGN",
            })
          }
        />

        <Line
          type="monotone"
          dataKey="predicted_price"
          stroke="#1f77b4"
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
