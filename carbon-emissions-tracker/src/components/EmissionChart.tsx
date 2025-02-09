import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface ChartProps {
  data: any[];
  stateName: string;
}

const EmissionsChart: React.FC<ChartProps> = ({ data }) => {
  // Reformat data to match Recharts requirements by extracting year and emissions
  const formattedData = data
    .map((item) => {
      // Extract emissions data for each year
      const emissionsData = Object.keys(item)
        .filter((key) => !isNaN(Number(key))) // Year keys should be numeric
        .map((year) => ({
          year,
          emission: item[year],
        }));

      return emissionsData;
    })
    .flat();

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={formattedData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="year"
          label={{ value: "Year", position: "insideBottomRight", offset: -5 }}
        />
        <YAxis
          label={{
            value: "Emissions (metric tons)",
            angle: -90,
            position: "insideLeft",
          }}
        />
        <Tooltip />
        <Legend />
        <Line
          type="monotone"
          dataKey="emission"
          stroke="#8884d8"
          activeDot={{ r: 8 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default EmissionsChart;
