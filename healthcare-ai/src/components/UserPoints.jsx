import React, { useState, useEffect, useContext } from "react";
import axios from "axios";
import { AuthContext } from "../context/AuthContext";
import { motion } from "framer-motion";

const UserPoints = () => {
  const { user } = useContext(AuthContext);
  const [points, setPoints] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchPoints = async () => {
      try {
        const response = await axios.get("http://localhost:5000/user/points", {
          headers: { "x-access-token": user.token },
        });
        setPoints(response.data.points);
        setLoading(false);
      } catch (err) {
        setError("Failed to fetch points");
        setLoading(false);
      }
    };

    if (user && user.token) {
      fetchPoints();
    }
  }, [user]);

  if (!user) {
    return (
      <div className="text-center text-[#4b5563] mt-8">
        Please log in to view your points.
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8 }}
      className="min-h-screen bg-gradient-to-b from-[#e0f2fe] to-[#bfdbfe] flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8"
    >
      <div className="max-w-md w-full bg-white rounded-2xl shadow-xl p-8">
        <h2 className="text-3xl font-bold text-center text-[#1e3a8a] mb-6">
          Your Reward Points
        </h2>
        {loading ? (
          <p className="text-center text-[#4b5563]">Loading...</p>
        ) : error ? (
          <p className="text-center text-red-500">{error}</p>
        ) : (
          <p className="text-center text-[#1e3a8a] text-lg">
            You have <span className="font-bold">{points}</span> reward points!
          </p>
        )}
        <p className="text-center text-[#4b5563] mt-4">
          Earn more points by submitting valid water issue reports.
        </p>
      </div>
    </motion.div>
  );
};

export default UserPoints;