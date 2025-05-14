import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { FaMapMarkerAlt, FaImage } from "react-icons/fa";

const ResolvedReport = () => {
  const [reports, setReports] = useState([]);
  const [token, setToken] = useState(localStorage.getItem("token"));

  useEffect(() => {
    fetchResolvedReports();
  }, []);

  const fetchResolvedReports = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/officer/resolved_reports", {
        headers: {
          "x-access-token": token,
        },
      });
      const data = await response.json();
      console.log("Fetched resolved reports:", data);
      setReports(data);
    } catch (error) {
      console.error("Error fetching resolved reports:", error);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-100 via-blue-200 to-purple-200 p-6 overflow-hidden">
      <motion.header
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, type: "spring" }}
        className="text-center mb-8"
      >
        <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-teal-500 to-purple-600">
          Resolved Reports
        </h1>
        <p className="text-gray-700 mt-2 italic">“Celebrating Our Success in Restoring Waters!”</p>
      </motion.header>

      <div className="max-w-5xl mx-auto">
        {reports.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center p-8 bg-white/80 backdrop-blur-md rounded-xl shadow-lg"
          >
            <FaImage className="text-5xl text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600 text-lg">No resolved reports yet.</p>
          </motion.div>
        ) : (
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {reports.map((report, index) => (
              <motion.div
                key={report._id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="p-5 bg-white/90 backdrop-blur-md rounded-xl shadow-lg border border-teal-100 hover:shadow-xl transition-shadow"
              >
                <div className="flex items-center gap-3 mb-3">
                  <FaMapMarkerAlt className="text-teal-500 text-2xl" />
                  <h2 className="text-xl font-semibold text-teal-700 truncate">{report.address}</h2>
                </div>

                <div className="space-y-2 text-gray-700">
                  <p>
                    <span className="font-medium">Issue:</span>{" "}
                    <span className={report.status === "leakage" ? "text-blue-500" : report.status === "pollution" ? "text-red-500" : "text-yellow-500"}>{report.status}</span>
                  </p>
                  <p>
                    <span className="font-medium">Confidence:</span>{" "}
                    {(report.confidence ? report.confidence * 100 : 0).toFixed(2)}%
                  </p>
                  <p>
                    <span className="font-medium">Assigned Officer:</span> {report.assigned_officer}
                  </p>
                  <p>
                    <span className="font-medium">Contact:</span> {report.officer_email}, {report.officer_phone}
                  </p>
                  <div>
                    <span className="font-medium">Original Image:</span>{" "}
                    {report.image ? (
                      <div className="mt-2">
                        <img
                          src={report.image}
                          alt="Water issue"
                          className="w-full h-48 object-cover rounded-md"
                          onError={(e) => {
                            console.error("Failed to load original image:", report.image);
                            e.target.src = "https://via.placeholder.com/150?text=Image+Not+Found";
                          }}
                          onLoad={() => console.log("Original image loaded successfully:", report.image)}
                        />
                        <a
                          href={report.image}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-teal-500 hover:underline text-sm mt-1 inline-block"
                        >
                          Open in new tab
                        </a>
                      </div>
                    ) : (
                      <span>No Image</span>
                    )}
                  </div>
                  <div>
                    <span className="font-medium">Resolved Image:</span>{" "}
                    {report.resolved_image ? (
                      <div className="mt-2">
                        <img
                          src={report.resolved_image}
                          alt="Resolved issue"
                          className="w-full h-48 object-cover rounded-md"
                          onError={(e) => {
                            console.error("Failed to load resolved image:", report.resolved_image);
                            e.target.src = "https://via.placeholder.com/150?text=Image+Not+Found";
                          }}
                          onLoad={() => console.log("Resolved image loaded successfully:", report.resolved_image)}
                        />
                        <a
                          href={report.resolved_image}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-teal-500 hover:underline text-sm mt-1 inline-block"
                        >
                          Open in new tab
                        </a>
                      </div>
                    ) : (
                      <span>No Resolved Image</span>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>

      <footer className="mt-8 text-center text-gray-600">
        <p className="font-semibold">“Together, We’ve Made a Difference!”</p>
      </footer>
    </div>
  );
};

export default ResolvedReport;