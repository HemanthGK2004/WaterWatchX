import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
} from "recharts";
import { FaSearch, FaFilter, FaDownload } from "react-icons/fa";
import { MdExpandMore } from "react-icons/md";
import axios from "axios";

const COLORS = ["#22C55E", "#FF5C5C"];

const Report = () => {
  const [reports, setReports] = useState([]);
  const [resolvedReports, setResolvedReports] = useState([]);
  const [filteredReports, setFilteredReports] = useState([]);
  const [search, setSearch] = useState("");
  const [showFilters, setShowFilters] = useState(false);
  const [dateRange, setDateRange] = useState("all");
  const [selectedReport, setSelectedReport] = useState(null);

  useEffect(() => {
    // Fetch reports from Flask backend
    axios.get("http://localhost:5000/get_reports")
      .then(response => {
        const data = response.data;
        const resolved = data.filter(report => report.resolved);
        setReports(data);
        setResolvedReports(resolved);
        setFilteredReports(resolved);
      })
      .catch(error => {
        console.error("Error fetching reports:", error);
      });
  }, []);

  useEffect(() => {
    let filtered = resolvedReports.filter(
      (report) =>
        (report.address?.toLowerCase().includes(search.toLowerCase()) ||
          report.assigned_officer?.toLowerCase().includes(search.toLowerCase())) &&
        (dateRange === "all" ||
         (dateRange === "week" && isWithinWeek(report.created_at)) ||
         (dateRange === "month" && isWithinMonth(report.created_at)))
    );
    setFilteredReports(filtered);
  }, [search, resolvedReports, dateRange]);

  const isWithinWeek = (date) => {
    const now = new Date();
    const reportDate = new Date(date);
    return (now - reportDate) / (1000 * 60 * 60 * 24) <= 7;
  };

  const isWithinMonth = (date) => {
    const now = new Date();
    const reportDate = new Date(date);
    return (now - reportDate) / (1000 * 60 * 60 * 24) <= 30;
  };

  const totalReports = reports.length;
  const totalResolved = resolvedReports.length;
  const pendingReports = totalReports - totalResolved;

  const pieData = [
    { name: "Resolved Reports", value: totalResolved },
    { name: "Pending Reports", value: pendingReports },
  ];

  const barData = [
    { name: "Total Reports", count: totalReports },
    { name: "Resolved Reports", count: totalResolved },
  ];

  const trendData = resolvedReports.reduce((acc, report) => {
    const date = new Date(report.created_at).toLocaleDateString();
    acc[date] = (acc[date] || 0) + 1;
    return acc;
  }, {});

  const lineData = Object.entries(trendData).map(([date, count]) => ({
    date,
    count,
  }));

  const downloadCSV = () => {
    const csv = [
      ["Address", "Latitude", "Longitude", "Officer", "Created Date"],
      ...filteredReports.map(r => [
        r.address || "N/A",
        r.latitude || "N/A",
        r.longitude || "N/A",
        r.assigned_officer || "N/A",
        r.created_at ? new Date(r.created_at).toLocaleDateString() : "N/A"
      ])
    ].map(row => row.join(",")).join("\n");
    
    const blob = new Blob([csv], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "resolved_reports.csv";
    a.click();
  };

  return (
    <div className="flex flex-col items-center min-h-screen bg-gradient-to-br from-green-100 via-blue-100 to-purple-100 p-6">
      {/* Enhanced Header */}
      <motion.div 
        className="text-center mb-8"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6 }}
      >
        <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-green-600 to-blue-600">
          AI-Based Water Issue Reports Dashboard
        </h1>
        <p className="text-gray-600 italic mt-2 text-lg">
          "Leveraging AI to detect and resolve water problems for a sustainable future"
        </p>
      </motion.div>

      {/* Search and Filter Section */}
      <div className="w-full max-w-4xl flex flex-col gap-4 mb-8">
        <div className="flex items-center gap-3">
          <input
            type="text"
            placeholder="Search by location or officer..."
            className="w-full p-3 rounded-full bg-white/80 backdrop-blur-sm border border-green-300 focus:ring-2 focus:ring-green-500 shadow-md"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <button className="bg-gradient-to-r from-green-500 to-blue-500 text-white p-3 rounded-full hover:scale-105 transition-transform">
            <FaSearch />
          </button>
          <button 
            onClick={() => setShowFilters(!showFilters)}
            className="bg-gray-200 p-3 rounded-full hover:bg-gray-300 transition-colors"
          >
            <FaFilter />
          </button>
          <button 
            onClick={downloadCSV}
            className="bg-blue-600 text-white p-3 rounded-full hover:bg-blue-700 transition-colors"
          >
            <FaDownload />
          </button>
        </div>

        <AnimatePresence>
          {showFilters && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="bg-white/90 backdrop-blur-sm p-4 rounded-lg shadow-lg"
            >
              <h3 className="text-lg font-semibold mb-2">Filter by Date</h3>
              <div className="flex gap-4">
                {["all", "week", "month"].map((range) => (
                  <button
                    key={range}
                    onClick={() => setDateRange(range)}
                    className={`px-4 py-2 rounded-full capitalize ${
                      dateRange === range 
                        ? "bg-green-500 text-white" 
                        : "bg-gray-200 hover:bg-gray-300"
                    }`}
                  >
                    {range}
                  </button>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Enhanced Charts Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-6xl">
        {/* Pie Chart */}
        <motion.div 
          className="bg-white/90 backdrop-blur-sm shadow-xl rounded-2xl p-6"
          whileHover={{ scale: 1.02 }}
        >
          <h2 className="text-xl font-semibold text-center text-gray-700 mb-4">
            Report Status
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={70}
                outerRadius={110}
                fill="#82ca9d"
                paddingAngle={5}
                dataKey="value"
                label={({ name, percent }) =>
                  `${name} ${(percent * 100).toFixed(0)}%`
                }
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Bar Chart */}
        <motion.div 
          className="bg-white/90 backdrop-blur-sm shadow-xl rounded-2xl p-6"
          whileHover={{ scale: 1.02 }}
        >
          <h2 className="text-xl font-semibold text-center text-gray-700 mb-4">
            Reports Comparison
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={barData} barSize={60}>
              <XAxis dataKey="name" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Legend />
              <Bar 
                dataKey="count" 
                fill="rgba(34, 197, 94, 0.8)" 
                radius={[10, 10, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Line Chart */}
        <motion.div 
          className="bg-white/90 backdrop-blur-sm shadow-xl rounded-2xl p-6"
          whileHover={{ scale: 1.02 }}
        >
          <h2 className="text-xl font-semibold text-center text-gray-700 mb-4">
            Resolution Trend
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={lineData}>
              <XAxis dataKey="date" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="count" 
                stroke="#22C55E" 
                strokeWidth={2}
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Enhanced Table with Expandable Rows */}
      <div className="w-full max-w-6xl bg-white/90 backdrop-blur-smIX shadow-xl rounded-2xl p-6 mt-8">
        {filteredReports.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-gradient-to-r from-green-500 to-blue-500 text-white text-lg">
                  <th className="px-6 py-3 text-left">Image</th>
                  <th className="px-6 py-3 text-left">Address</th>
                  <th className="px-6 py-3 text-left">Latitude</th>
                  <th className="px-6 py-3 text-left">Longitude</th>
                  <th className="px-6 py-3 text-left">Officer</th>
                  <th className="px-6 py-3 text-left">Created Date</th>
                  <th className="px-6 py-3 text-left"></th>
                </tr>
              </thead>
              <tbody>
                {filteredReports.map((report, index) => (
                  <React.Fragment key={index}>
                    <motion.tr
                      className="bg-gray-50 hover:bg-green-50 transition-colors"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: index * 0.1 }}
                      onClick={() => setSelectedReport(selectedReport === index ? null : index)}
                    >
                      <td className="border px-6 py-4">
                        <img
                          src={report.resolved_image || report.image}
                          alt="Resolved Report"
                          className="w-16 h-16 object-cover rounded-md shadow-sm"
                          onError={(e) => {
                            e.target.src = "/placeholder.png";
                          }}
                        />
                      </td>
                      <td className="border px-6 py-4">{report.address || "N/A"}</td>
                      <td className="border px-6 py-4">{report.latitude || "N/A"}</td>
                      <td className="border px-6 py-4">{report.longitude || "N/A"}</td>
                      <td className="border px-6 py-4 font-semibold text-green-700">
                        {report.assigned_officer || "N/A"}
                      </td>
                      <td className="border px-6 py-4 text-gray-500">
                        {report.created_at ? new Date(report.created_at).toLocaleDateString() : "N/A"}
                      </td>
                      <td className="border px-6 py-4">
                        <MdExpandMore className={`transition-transform ${selectedReport === index ? "rotate-180" : ""}`} />
                      </td>
                    </motion.tr>
                    <AnimatePresence>
                      {selectedReport === index && (
                        <motion.tr
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: "auto", opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                        >
                          <td colSpan="7" className="p-4 bg-gray-100">
                            <div className="flex gap-4">
                              <img
                                src={report.resolved_image || report.image}
                                alt="Expanded view"
                                className="w-32 h-32 object-cover rounded-lg shadow-md"
                              />
                              <div>
                                <p><strong>Description:</strong> {report.status || "No description available"}</p>
                                <p><strong>Status:</strong> {report.resolved ? "Resolved" : "Pending"}</p>
                                <button className="mt-2 px-4 py-2 bg-green-500 text-white rounded-full hover:bg-green-600">
                                  View Full Details
                                </button>
                              </div>
                            </div>
                          </td>
                        </motion.tr>
                      )}
                    </AnimatePresence>
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-center text-gray-500 text-lg mt-4 animate-pulse">
            No resolved reports found.
          </p>
        )}
      </div>
    </div>
  );
};

export default Report;