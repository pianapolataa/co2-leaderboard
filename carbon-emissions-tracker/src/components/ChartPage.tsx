import React, { useState, useMemo } from "react";
import {
  Box,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Paper,
  Grid,
  Card,
  CardContent,
  IconButton,
  Tooltip as MuiTooltip,
  useTheme,
  CircularProgress,
  Divider,
  Button,
} from "@mui/material";
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
import { Link } from "react-router-dom";
import InfoIcon from "@mui/icons-material/Info";
import TrendingUpIcon from "@mui/icons-material/TrendingUp";
import TrendingDownIcon from "@mui/icons-material/TrendingDown";
import HomeIcon from "@mui/icons-material/Home";
import MenuBookIcon from "@mui/icons-material/MenuBook";

interface ChartPageProps {
  data: Array<{
    State: string;
    [key: string]: number | string;
  }>;
}

const ChartPage: React.FC<ChartPageProps> = ({ data }) => {
  const theme = useTheme();
  const [selectedState, setSelectedState] = useState<string>("California");

  const chartData = useMemo(() => {
    const stateData = data.find((item) => item.State === selectedState);
    if (!stateData) return [];

    return Object.entries(stateData)
      .filter(([key]) => !isNaN(Number(key)))
      .map(([year, emission]) => ({
        year,
        emission: Number(emission),
      }))
      .sort((a, b) => Number(a.year) - Number(b.year));
  }, [data, selectedState]);

  const statistics = useMemo(() => {
    if (chartData.length === 0) return null;

    const latestYear = chartData[chartData.length - 1];
    const previousYear = chartData[chartData.length - 2];
    const firstYear = chartData[0];

    const yearOverYearChange = previousYear
      ? ((latestYear.emission - previousYear.emission) /
          previousYear.emission) *
        100
      : 0;

    const totalChange = firstYear
      ? ((latestYear.emission - firstYear.emission) / firstYear.emission) * 100
      : 0;

    return {
      current: latestYear.emission,
      yearOverYearChange,
      totalChange,
      yearRange: `${firstYear.year} - ${latestYear.year}`,
    };
  }, [chartData]);

  if (!data.length) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        height="100vh"
      >
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ padding: { xs: 2, md: 4 }, maxWidth: 1400, margin: "0 auto" }}>
      <Grid container spacing={3}>
        {/* Header Section */}
        <Grid item xs={12}>
          <Typography
            variant="h4"
            gutterBottom
            sx={{
              fontWeight: "bold",
              color: theme.palette.primary.main,
              textAlign: "center",
            }}
          >
            Emissions Trends Analysis
          </Typography>
          <Divider sx={{ marginBottom: 3 }} />
        </Grid>

        {/* State Selection */}
        <Grid item xs={12} md={4}>
          <FormControl fullWidth>
            <InputLabel>Select State</InputLabel>
            <Select
              value={selectedState}
              onChange={(e) => setSelectedState(e.target.value)}
              label="Select State"
              sx={{ backgroundColor: "white" }}
            >
              {data.map((item) => (
                <MenuItem key={item.State} value={item.State}>
                  {item.State}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        {/* Statistics Summary */}
        {statistics && (
          <Grid item xs={12}>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={4}>
                <StatCard
                  title="Current Emissions"
                  value={`${statistics.current.toLocaleString()} units`}
                  subtitle={`Latest year: ${
                    chartData[chartData.length - 1].year
                  }`}
                />
              </Grid>
              <Grid item xs={12} sm={4}>
                <StatCard
                  title="Year-over-Year Change"
                  value={`${statistics.yearOverYearChange.toFixed(1)}%`}
                  subtitle="Previous year comparison"
                  trend={statistics.yearOverYearChange}
                />
              </Grid>
              <Grid item xs={12} sm={4}>
                <StatCard
                  title="Total Change"
                  value={`${statistics.totalChange.toFixed(1)}%`}
                  subtitle={`Period: ${statistics.yearRange}`}
                  trend={statistics.totalChange}
                />
              </Grid>
            </Grid>
          </Grid>
        )}

        {/* Chart Section */}
        <Grid item xs={12}>
          <Paper
            elevation={3}
            sx={{
              p: 3,
              borderRadius: 2,
              backgroundColor: "white",
            }}
          >
            <Typography
              variant="h6"
              gutterBottom
              sx={{ display: "flex", alignItems: "center" }}
            >
              Emissions Timeline for {selectedState}
              <MuiTooltip title="Shows annual emissions trends over time">
                <IconButton size="small" sx={{ ml: 1 }}>
                  <InfoIcon fontSize="small" />
                </IconButton>
              </MuiTooltip>
            </Typography>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={chartData}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke={theme.palette.grey[200]}
                />
                <XAxis
                  dataKey="year"
                  tick={{ fill: theme.palette.text.secondary }}
                />
                <YAxis
                  tick={{ fill: theme.palette.text.secondary }}
                  label={{
                    value: "Emissions (units)",
                    angle: -90,
                    position: "insideLeft",
                    style: { fill: theme.palette.text.secondary },
                  }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: theme.palette.background.paper,
                    border: `1px solid ${theme.palette.divider}`,
                    borderRadius: theme.shape.borderRadius,
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="emission"
                  name="Emissions"
                  stroke={theme.palette.primary.main}
                  strokeWidth={3}
                  dot={{ fill: theme.palette.primary.main }}
                  activeDot={{ r: 8 }}
                  animationDuration={1000}
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Learn More Section */}
        <Grid item xs={12}>
          <Paper
            elevation={3}
            sx={{
              p: 4,
              mt: 3,
              backgroundColor: theme.palette.background.default,
              textAlign: "left",
            }}
          >
            <Typography variant="h5" gutterBottom>
              About the Paris Agreement
            </Typography>
            <Typography variant="body1" paragraph>
              The Paris Agreement is an international treaty that was adopted by
              196 countries in December 2015 to combat climate change and limit
              global warming to below 2°C, preferably to 1.5°C, compared to
              pre-industrial levels. It emphasizes reducing greenhouse gas
              emissions and adapting to the impacts of climate change.
            </Typography>
            <Typography variant="body1" paragraph>
              Participating countries have set nationally determined
              contributions (NDCs) to help achieve these goals. Monitoring and
              reporting mechanisms ensure transparency and progress tracking.
              This global effort supports sustainable development and improved
              resilience to climate challenges.
            </Typography>
            <Button
              variant="contained"
              color="primary"
              startIcon={<MenuBookIcon />}
              href="https://unfccc.int/process-and-meetings/the-paris-agreement"
              target="_blank"
              sx={{ mt: 2 }}
            >
              Learn More
            </Button>
          </Paper>
        </Grid>

        {/* Home Link */}
        <Grid item xs={12} sx={{ textAlign: "center", mt: 2 }}>
          <Link to="/" style={{ textDecoration: "none" }}>
            <IconButton color="primary" size="large">
              <HomeIcon />
            </IconButton>
          </Link>
        </Grid>
      </Grid>
    </Box>
  );
};

interface StatCardProps {
  title: string;
  value: string;
  subtitle: string;
  trend?: number;
}

const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  subtitle,
  trend,
}) => {
  const theme = useTheme();

  return (
    <Card
      sx={{ height: "100%", backgroundColor: theme.palette.background.paper }}
    >
      <CardContent>
        <Typography variant="subtitle2" color="textSecondary">
          {title}
        </Typography>
        <Typography
          variant="h5"
          sx={{
            mt: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color:
              trend !== undefined
                ? trend > 0
                  ? theme.palette.error.main
                  : theme.palette.success.main
                : "inherit",
          }}
        >
          {value}
          {trend !== undefined &&
            (trend > 0 ? (
              <TrendingUpIcon sx={{ ml: 1 }} />
            ) : (
              <TrendingDownIcon sx={{ ml: 1 }} />
            ))}
        </Typography>
        <Typography variant="caption" color="textSecondary">
          {subtitle}
        </Typography>
      </CardContent>
    </Card>
  );
};

export default ChartPage;
