import React, { useEffect, useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Dashboard from "./components/Dashboard";
import ChartPage from "./components/ChartPage";
import { getAllEmissions } from "./api/emissions"; // Replace with your actual API function

const App: React.FC = () => {
  const [data, setData] = useState<any[]>([]);

  useEffect(() => {
    getAllEmissions().then(setData); // Fetch data from your API
  }, []);

  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/charts" element={<ChartPage data={data} />} />
      </Routes>
    </Router>
  );
};

export default App;
