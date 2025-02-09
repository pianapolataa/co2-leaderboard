import React, { useState } from "react";
import { ComposableMap, Geographies, Geography } from "react-simple-maps";
import {
  Box,
  Typography,
  Grid,
  Paper,
  MenuItem,
  Select,
  Tooltip,
  FormControl,
  InputLabel,
  TextField,
  useMediaQuery,
  useTheme,
  Button,
  ButtonGroup,
  Chip,
  Slide,
  Fade,
  Divider,
  IconButton,
} from "@mui/material";
import * as d3 from "d3";
import { useNavigate } from "react-router-dom";
import { InfoRounded, Timeline, ArrowDropDown } from "@mui/icons-material";

const regions: Record<string, string[]> = {
  Northeast: ["New York", "Pennsylvania", "Massachusetts", "New Jersey"],
  Midwest: ["Illinois", "Ohio", "Michigan", "Indiana"],
  South: ["Texas", "Florida", "Georgia", "North Carolina"],
  West: ["California", "Washington", "Oregon", "Nevada"],
};

const geoUrl = "https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json";

interface USMapProps {
  data: any[];
}

const USMap: React.FC<USMapProps> = ({ data }) => {
  const theme = useTheme();
  const isSmallScreen = useMediaQuery(theme.breakpoints.down("md"));
  const navigate = useNavigate();

  const [year, setYear] = useState<string>("2022");
  const [region, setRegion] = useState<string>("All");
  const [emissionRange, setEmissionRange] = useState<[number, number]>([
    0, 400,
  ]);
  const [tooltipContent, setTooltipContent] = useState("");
  const [showFilters, setShowFilters] = useState(true);
  const [interpolatorName, setInterpolatorName] = useState<string>("Cool");

  const interpolators: Record<string, (t: number) => string> = {
    Cool: d3.interpolateCool,
    Warm: d3.interpolateWarm,
    Viridis: d3.interpolateViridis,
    Plasma: d3.interpolatePlasma,
  };

  const colorScale = d3
    .scaleSequential(interpolators[interpolatorName])
    .domain([0, 400]);

  const getEmissionData = (stateName: string, selectedYear: string) => {
    const stateData = data.find(
      (item) => item.State.toLowerCase() === stateName.toLowerCase()
    );
    return stateData ? stateData[selectedYear] : null;
  };

  const filterStates = (
    stateName: string,
    emission: number | null
  ): boolean => {
    if (region !== "All" && !regions[region].includes(stateName)) return false;
    if (
      emission === null ||
      emission < emissionRange[0] ||
      emission > emissionRange[1]
    )
      return false;
    return true;
  };

  return (
    <Box
      sx={{
        width: "100%",
        padding: 2,
        marginTop: 8,
        background: theme.palette.background.default,
        minHeight: "100vh",
      }}
    >
      <Box
        sx={{
          maxWidth: 1440,
          margin: "0 auto",
          padding: theme.spacing(3),
        }}
      >
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            mb: 4,
          }}
        >
          <Typography
            variant="h3"
            sx={{
              fontWeight: 800,
              color: theme.palette.text.primary,
              letterSpacing: "-0.05rem",
            }}
          >
            Carbon Emissions Dashboard
            <Chip
              label="Beta"
              size="small"
              sx={{
                ml: 2,
                background: theme.palette.success.dark,
                color: "white",
                fontSize: "0.75rem",
              }}
            />
          </Typography>

          <IconButton onClick={() => setShowFilters(!showFilters)}>
            <ArrowDropDown
              sx={{
                transform: showFilters ? "rotate(180deg)" : "rotate(0)",
                transition: "transform 0.3s ease",
              }}
            />
          </IconButton>
        </Box>

        <Slide in={showFilters} direction="down">
          <Box
            sx={{
              background: theme.palette.background.paper,
              borderRadius: 4,
              p: 3,
              mb: 4,
              boxShadow: theme.shadows[3],
            }}
          >
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <FormControl fullWidth>
                  <InputLabel
                    sx={{
                      color: theme.palette.text.secondary,
                      "&.Mui-focused": {
                        color: theme.palette.primary.main,
                      },
                    }}
                  >
                    Year
                  </InputLabel>
                  <Select
                    value={year}
                    onChange={(e) => setYear(e.target.value)}
                    label="Year"
                    MenuProps={{
                      PaperProps: {
                        sx: {
                          borderRadius: 2,
                          marginTop: 1,
                          boxShadow: theme.shadows[3],
                        },
                      },
                    }}
                  >
                    {Object.keys(data[0] || {})
                      .filter(
                        (key) => !isNaN(Number(key)) && Number(key) <= 2022
                      )
                      .map((yearOption) => (
                        <MenuItem
                          key={yearOption}
                          value={yearOption}
                          sx={{
                            "&:hover": {
                              background: theme.palette.action.hover,
                            },
                          }}
                        >
                          {yearOption}
                        </MenuItem>
                      ))}
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={4}>
                <FormControl fullWidth>
                  <InputLabel
                    sx={{
                      color: theme.palette.text.secondary,
                      "&.Mui-focused": {
                        color: theme.palette.primary.main,
                      },
                    }}
                  >
                    Region
                  </InputLabel>
                  <Select
                    value={region}
                    onChange={(e) => setRegion(e.target.value)}
                    label="Region"
                  >
                    <MenuItem value="All">All Regions</MenuItem>
                    {Object.keys(regions).map((regionOption) => (
                      <MenuItem
                        key={regionOption}
                        value={regionOption}
                        sx={{
                          "&:hover": {
                            background: theme.palette.action.hover,
                          },
                        }}
                      >
                        {regionOption}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={4}>
                <Box
                  sx={{
                    display: "flex",
                    gap: 2,
                    "& .MuiTextField-root": {
                      flex: 1,
                    },
                  }}
                >
                  <TextField
                    label="Min Emission"
                    type="number"
                    value={emissionRange[0]}
                    onChange={(e) =>
                      setEmissionRange([+e.target.value, emissionRange[1]])
                    }
                    variant="outlined"
                    InputProps={{
                      endAdornment: (
                        <Typography variant="caption">tons</Typography>
                      ),
                    }}
                  />
                  <TextField
                    label="Max Emission"
                    type="number"
                    value={emissionRange[1]}
                    onChange={(e) =>
                      setEmissionRange([emissionRange[0], +e.target.value])
                    }
                    variant="outlined"
                    InputProps={{
                      endAdornment: (
                        <Typography variant="caption">tons</Typography>
                      ),
                    }}
                  />
                </Box>
              </Grid>
            </Grid>
          </Box>
        </Slide>

        <Grid container spacing={4}>
          <Grid item xs={12} lg={8}>
            <Box
              sx={{
                background: theme.palette.background.paper,
                borderRadius: 4,
                p: 3,
                boxShadow: theme.shadows[3],
                position: "relative",
              }}
            >
              <Box
                sx={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  mb: 3,
                }}
              >
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  Interactive Emissions Map
                </Typography>
                <Tooltip title="Data source: EPA Carbon Emissions Database">
                  <InfoRounded
                    sx={{
                      color: theme.palette.text.secondary,
                      fontSize: "1.2rem",
                    }}
                  />
                </Tooltip>
              </Box>

              <ComposableMap
                projection="geoAlbersUsa"
                style={{ width: "100%", height: "500px" }}
              >
                <Geographies geography={geoUrl}>
                  {({ geographies }) =>
                    geographies.map((geo) => {
                      const stateName = geo.properties.name;
                      const emission = getEmissionData(stateName, year);
                      if (!filterStates(stateName, emission)) return null;

                      return (
                        <Geography
                          key={geo.rsmKey}
                          geography={geo}
                          onMouseEnter={() =>
                            setTooltipContent(
                              `${stateName}: ${
                                emission !== null
                                  ? `${emission.toLocaleString()} metric tons`
                                  : "No data"
                              }`
                            )
                          }
                          onMouseLeave={() => setTooltipContent("")}
                          style={{
                            default: {
                              fill: emission ? colorScale(emission) : "#F5F5F5",
                              outline: "none",
                              stroke: theme.palette.divider,
                              strokeWidth: 0.5,
                            },
                            hover: {
                              fill: theme.palette.primary.main,
                              outline: "none",
                              cursor: "pointer",
                            },
                            pressed: {
                              fill: theme.palette.primary.dark,
                              outline: "none",
                            },
                          }}
                        />
                      );
                    })
                  }
                </Geographies>
              </ComposableMap>

              <Fade in={!!tooltipContent}>
                <Paper
                  sx={{
                    position: "absolute",
                    top: 20,
                    left: 20,
                    p: 2,
                    borderRadius: 2,
                    boxShadow: theme.shadows[2],
                    background: theme.palette.background.paper,
                  }}
                >
                  <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                    {tooltipContent}
                  </Typography>
                </Paper>
              </Fade>
            </Box>
          </Grid>

          <Grid item xs={12} lg={4}>
            <Box
              sx={{
                background: theme.palette.background.paper,
                borderRadius: 4,
                p: 3,
                height: "100%",
                boxShadow: theme.shadows[3],
              }}
            >
              <Typography
                variant="h6"
                sx={{
                  fontWeight: 600,
                  mb: 3,
                }}
              >
                Predictions & Insights
              </Typography>

              <Divider sx={{ mb: 3 }} />

              <ButtonGroup
                fullWidth
                orientation={isSmallScreen ? "horizontal" : "vertical"}
                sx={{ mb: 3 }}
              >
                {[2023, 2024, 2025, 2026, 2027].map((yearOption) => (
                  <Button
                    key={yearOption}
                    onClick={() => setYear(yearOption.toString())}
                    variant={
                      year === yearOption.toString() ? "contained" : "outlined"
                    }
                    sx={{
                      textTransform: "none",
                      justifyContent: "space-between",
                      py: 1.5,
                    }}
                  >
                    {yearOption} Projection
                    <Timeline
                      sx={{
                        fontSize: "1rem",
                        ml: 1,
                      }}
                    />
                  </Button>
                ))}
              </ButtonGroup>
              <Divider sx={{ mb: 3 }} />

              {/* New Button Group to choose the color interpolator */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  Choose Color Interpolator:
                </Typography>

                <ButtonGroup
                  fullWidth
                  orientation={isSmallScreen ? "vertical" : "horizontal"}
                  sx={{ mb: 3 }}
                >
                  {["Cool", "Warm", "Viridis", "Plasma"].map((name) => (
                    <Button
                      key={name}
                      variant={
                        interpolatorName === name ? "contained" : "outlined"
                      }
                      onClick={() => setInterpolatorName(name)}
                    >
                      {name}
                    </Button>
                  ))}
                </ButtonGroup>
                <Divider sx={{ mb: 3 }} />
              </Box>

              <Box
                sx={{
                  background: theme.palette.action.hover,
                  borderRadius: 2,
                  p: 2,
                  mt: "auto",
                }}
              >
                <Typography
                  variant="body2"
                  sx={{
                    color: theme.palette.text.secondary,
                    mb: 1,
                  }}
                >
                  Color Legend (metric tons)
                </Typography>
                <Box
                  sx={{
                    height: 20,
                    background: `linear-gradient(to right, ${d3
                      .range(0, 400, 50)
                      .map((d) => colorScale(d))
                      .join(", ")})`,
                    borderRadius: 1,
                    mb: 1,
                  }}
                />
                <Box
                  sx={{
                    display: "flex",
                    justifyContent: "space-between",
                  }}
                >
                  {[0, 100, 200, 300, 400].map((value) => (
                    <Typography
                      key={value}
                      variant="caption"
                      sx={{ color: theme.palette.text.secondary }}
                    >
                      {value}
                    </Typography>
                  ))}
                </Box>
              </Box>
            </Box>
          </Grid>
        </Grid>

        <Box
          sx={{
            mt: 4,
            textAlign: "center",
            display: "flex",
            justifyContent: "center",
            gap: 2,
          }}
        >
          <Button
            variant="contained"
            color="primary"
            onClick={() => navigate("/charts")}
            sx={{
              px: 4,
              py: 1.5,
              borderRadius: 2,
              fontWeight: 600,
            }}
          >
            Advanced Analytics
            <Timeline sx={{ ml: 1.5 }} />
          </Button>
        </Box>
      </Box>
    </Box>
  );
};

export default USMap;
