/* eslint-disable */

import React, { useEffect, useState } from "react";
import Card from "./Card";
import { useCurrTab } from "./CurrTabContext";
import { useFetch } from "./useFetch";
import Loader from "./Loader";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function Overview() {
  const { currState, currCommodity } = useCurrTab();
  const { data: overviewData, loading } = useFetch(
    `/historical_prices/${currCommodity}/${currState}`
  );

  const [stats, setStats] = useState([]);

  useEffect(() => {
    if (overviewData) {
      console.log(overviewData);
      setStats(
        Object.entries(overviewData.statistics)
          .map(([stat, value]) => ({
            stat,
            value,
          }))
          .slice(0, -2)
      );
    }
  }, [overviewData]);

  return (
    <div className="pb-[100px]">
      {/* Heading */}
      <div>
        <h1 className="heading">
          {currCommodity} Prices in {currState}
        </h1>
        <p className="leading-[80%] mt-[10px] text-[18px]">
          An overview of historical and current price data (2024-11 to 2025-10)
        </p>
      </div>

      {loading ? (
        <div className="mt-[200px] justify-center items-center flex">
          <Loader size={100} />
        </div>
      ) : (
        <>
          {" "}
          {/* Container for stat cards */}
          <div className="flex flex-wrap justify-between items-center gap-[20px] mt-[60px]">
            {stats.map((stat) => {
              return <OverviewCard {...stat} />;
            })}
          </div>
          {/* Container for graph of historical Prices */}
          <Card>
            <p className="font-medium text-[20px] mt-[40px] p-[20px]">
              Historical Price Trend
            </p>

            <SingleLineChart data={overviewData?.data} />
          </Card>
        </>
      )}
    </div>
  );
}

function formatKey(text) {
  return text
    .replace(/_/g, " ") // replace underscores with spaces
    .replace(/\b\w/g, (c) => c.toUpperCase()); // capitalize each word
}

function OverviewCard({ stat, value }) {
  return (
    <Card className="px-[50px] p-[20px]">
      {/* Title */}
      <p className="font-medium text-[18px]">{formatKey(stat)}</p>

      {/* Value */}
      <p className="font-semibold text-[36px] mt-[10px]">
        {typeof value == "object"
          ? formatCurrency(value.price)
          : formatCurrency(value)}
      </p>
    </Card>
  );
}

export function formatCurrency(amount, round = 0) {
  return amount.toLocaleString("en-NG", {
    style: "currency",
    currency: "NGN",
    minimumFractionDigits: round,
    maximumFractionDigits: round,
  });
}

function formatLargeNumber(num) {
  let value, suffix;

  if (num >= 1_000_000_000) {
    value = num / 1_000_000_000;
    suffix = "B";
  } else if (num >= 1_000_000) {
    value = num / 1_000_000;
    suffix = "M";
  } else if (num >= 1_000) {
    value = num / 1_000;
    suffix = "K";
  } else {
    value = num;
    suffix = "";
  }

  // Round to nearest whole number, limit to 3 digits max (e.g. 999K+ before switching to M+)
  const rounded = Math.round(value);

  return `₦${value}${suffix}`;
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
          dataKey="price"
          stroke="#1f77b4"
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
