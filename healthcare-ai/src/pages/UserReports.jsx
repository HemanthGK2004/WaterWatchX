import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { useAuth } from "../context/AuthContext";

const UserReports = () => {
  const { isLoggedIn } = useAuth();
  const [reports, setReports] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!isLoggedIn) {
      setError("Please log in to view your reports.");
      setLoading(false);
      return;
    }

    const fetchReports = async () => {
      try {
        const token = localStorage.getItem("token");
        const response = await fetch("http://localhost:5000/user/reports", {
          method: "GET",
          headers: {
            "x-access-token": token,
          },
        });

        const data = await response.json();

        if (response.ok) {
          setReports(data);
        } else {
          setError(data.error || "Failed to fetch reports.");
        }
      } catch (err) {
        setError("Server error. Please try again later.");
        console.error("Fetch reports error:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchReports();
  }, [isLoggedIn]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-teal-300 via-blue-400 to-purple-500">
        <p className="text-white text-xl">Loading...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-300 via-blue-400 to-purple-500 py-12 px-6">
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="max-w-6xl mx-auto bg-white shadow-2xl rounded-2xl p-8"
      >
        <h1 className="text-4xl font-extrabold text-teal-600 mb-8 text-center">
          My Water Issue Reports
        </h1>

        {error && (
          <p className="text-red-500 bg-red-100 p-3 rounded-lg mb-6 text-center font-medium">
            {error}
          </p>
        )}

        {reports.length === 0 && !error ? (
          <p className="text-gray-600 text-center text-lg">
            You have not submitted any reports yet.
          </p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {reports.map((report) => (
              <motion.div
                key={report._id}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
                className="bg-gray-100 p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow"
              >
                <img
                  src={report.image}
                  alt="Water Issue"
                  className="w-full h-48 object-cover rounded-md mb-4"
                />
                <h2 className="text-xl font-semibold text-teal-600 mb-2">
                  {report.status.charAt(0).toUpperCase() + report.status.slice(1)}
                </h2>
                <p className="text-gray-700 mb-2">
                  <strong>Address:</strong> {report.address}
                </p>
                <p className="text-gray-700 mb-2">
                  <strong>Confidence:</strong> {(report.confidence * 100).toFixed(2)}%
                </p>
                <p className="text-gray-700 mb-2">
                  <strong>Assigned Officer:</strong> {report.assigned_officer}
                </p>
                <p className="text-gray-700 mb-2">
                  <strong>Officer Email:</strong> {report.officer_email || "N/A"}
                </p>
                <p className="text-gray-700 mb-2">
                  <strong>Officer Phone:</strong> {report.officer_phone || "N/A"}
                </p>
                <p className="text-gray-700">
                  <strong>Submitted:</strong>{" "}
                  {new Date(report.created_at).toLocaleDateString()}
                </p>
              </motion.div>
            ))}
          </div>
        )}
      </motion.div>
    </div>
  );
};

export default UserReports;