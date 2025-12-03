import React, { useEffect } from "react";
import { useCurrTab } from "./CurrTabContext";
import Card from "./Card";
import { useFetch } from "./useFetch";
import Loader from "./Loader";
import { formatCurrency } from "./Overview";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

export default function Analysis() {
  const { currState, currCommodity } = useCurrTab();
  //

  const { data: analyticsData, loading } = useFetch(
    `/analytics/${currState}/${currCommodity}`
  );

  useEffect(() => {
    if (analyticsData) {
      console.log(analyticsData);
    }
  }, [analyticsData]);

  return (
    <div>
      <h1 className="heading">
        Prediction Analysis for {currCommodity} in {currState}
      </h1>

      <p className="mt-[10px] text-[18px]">
        Model performance using statistical metrics for the year 2025
      </p>

      {loading ? (
        <div className="flex justify-center items-center mt-[200px]">
          <Loader size={100} />
        </div>
      ) : !analyticsData.error ? (
        <>
          {" "}
          {/* Container for statistcal cards */}
          <div className="flex gap-[30px] justify-start items-center mt-[50px]">
            <AnalysisCard
              stat="Average Price"
              value={formatCurrency(analyticsData?.mean_price)}
            />
            <AnalysisCard
              stat="Mean Absolute Error"
              value={formatCurrency(analyticsData?.mae)}
            />
            <AnalysisCard
              stat="Mean Absolute Percentage Error"
              value={`${analyticsData?.mape_pct.toFixed(2)}%`}
            />
            <AnalysisCard
              stat="Root Mean Squared Error"
              value={formatCurrency(analyticsData?.rmse)}
            />
            <AnalysisCard
              stat="Number of Observations"
              value={analyticsData?.n}
            />
          </div>
          {/* Card for graph comparing predicted prices vs actual prices in 2025 */}
          <Card className="mt-[40px] p-[20px]">
            <p className="text-[18px] font-medium ">
              Actual and Predicted Price Trend for 2025
            </p>

            <ActualVsPredictedChart data={analyticsData?.graph} />
          </Card>
        </>
      ) : (
        <>
          <p className="mt-[40px] text-[28px] font-medium">
            Not enough data for this pair
          </p>
        </>
      )}
    </div>
  );
}

function AnalysisCard({ stat, value, money }) {
  return (
    <Card className="flex flex-col justify-start items-start gap-[5px] px-[25px] py-[30px]">
      {/* Stat */}
      <p className="text-[16px] font-medium">{stat}</p>

      {/* Value */}
      <p className="text-[28px] font-semibold">
        {money && "₦"}
        {value}
      </p>
    </Card>
  );
}

// currency formatter for tooltip and Y-axis
const nairaFormatter = (value) =>
  Number(value).toLocaleString("en-NG", {
    style: "currency",
    currency: "NGN",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  });

function ActualVsPredictedChart({ data }) {
  // data expected: [{ month: "Jan 2025", actual_price: 514.88, predicted_price: 392.52 }, ...]

  // Optional: if month labels are numeric or you want to ensure order, you can sort here:
  const chartData = [...(data || [])];

  return (
    <ResponsiveContainer width="100%" height={360}>
      <LineChart
        data={chartData}
        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" />

        <XAxis
          dataKey="month"
          tick={{ fontSize: 12 }}
          interval={0}
          angle={0}
          textAnchor="middle"
        />

        <YAxis
          tickFormatter={(v) => nairaFormatter(v)}
          width={90}
          // optionally provide ticks if you want specific spacing:
          // ticks={[350, 400, 450, 500]}
        />

        <Tooltip
          formatter={(value, name) => [
            nairaFormatter(value),
            name === "actual_price" ? "Actual" : "Predicted",
          ]}
        />

        <Legend verticalAlign="top" height={36} />

        <Line
          type="monotone"
          dataKey="actual_price"
          name="Actual"
          stroke="#1f77b4"
          strokeWidth={2}
          dot={false}
        />

        <Line
          type="monotone"
          dataKey="predicted_price"
          name="Predicted"
          stroke="#ff7f0e"
          strokeWidth={2}
          strokeDasharray="5 3"
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
