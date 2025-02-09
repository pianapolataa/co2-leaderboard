import React, { useEffect, useState } from "react";
import { getAllEmissions } from "../api/emissions";
import { Container, CircularProgress } from "@mui/material";
import USMap from "./USMap";

const Dashboard: React.FC = () => {
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getAllEmissions().then((data) => {
      setData(data);
      setLoading(false);
    });
  }, []);

  return (
    <Container>
      {loading ? <CircularProgress /> : <USMap data={data} />}
    </Container>
  );
};

export default Dashboard;
