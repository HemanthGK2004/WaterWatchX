import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { motion } from "framer-motion";
import { FaUserLock, FaEnvelope, FaKey, FaSignInAlt, FaUserPlus, FaExclamationCircle } from "react-icons/fa";
import { useAuth } from "../context/AuthContext";

const Login = () => {
  const [identifier, setIdentifier] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [isOfficer, setIsOfficer] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();
  const { login } = useAuth();

  const handleLogin = async () => {
    setError(null);
    setIsLoading(true);
    if (!identifier || !password) {
      setError("Please fill in both fields.");
      setIsLoading(false);
      return;
    }

    const apiUrl = isOfficer
      ? "http://127.0.0.1:5000/login"
      : "http://127.0.0.1:5000/user/login";

    try {
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ identifier, password }),
      });

      const data = await response.json();

      if (response.ok) {
        if (!data.token || !data.role) {
          setError("Invalid response from server: Missing token or role.");
          return;
        }
        login(data.token, data.role);
        navigate(data.role === "officer" ? "/officers-dashboard" : "/UserDashboard");
      } else {
        setError(data.error || "Login failed. Please check your credentials.");
      }
    } catch (err) {
      setError("Unable to connect to the server. Please try again later.");
      console.error("Login error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-teal-50 p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="bg-white shadow-xl rounded-3xl overflow-hidden w-full max-w-md"
      >
        {/* Header Section */}
        <div className="bg-gradient-to-r from-teal-500 to-blue-600 p-6 text-center">
          <div className="flex items-center justify-center space-x-3">
            <FaUserLock className="text-white text-3xl" />
            <h1 className="text-3xl font-bold text-white">Welcome Back!</h1>
          </div>
          <p className="mt-2 text-blue-100">
            {isOfficer 
              ? "Officer access to water management system" 
              : "Join us in creating cleaner water solutions"}
          </p>
        </div>

        {/* Main Content */}
        <div className="p-8">
          {/* Motivational Quote */}
          <div className="mb-6 text-center">
            <p className="text-gray-600 italic">
              "Every drop counts. Your reports help us protect our water resources."
            </p>
            <p className="mt-2 text-sm text-blue-500 font-medium flex items-center justify-center">
              <FaExclamationCircle className="mr-1" />
              Please login before reporting water issues
            </p>
          </div>

          {error && (
            <motion.div 
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-red-50 border-l-4 border-red-500 p-3 rounded mb-6"
            >
              <p className="text-red-700 flex items-center">
                <FaExclamationCircle className="mr-2" /> {error}
              </p>
            </motion.div>
          )}

          {/* Role Selector */}
          <div className="flex justify-center mb-6">
            <div className="inline-flex rounded-full bg-gray-100 p-1">
              <button
                onClick={() => setIsOfficer(false)}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                  !isOfficer ? "bg-teal-500 text-white shadow-md" : "text-gray-600 hover:bg-gray-200"
                }`}
              >
                User Login
              </button>
              <button
                onClick={() => setIsOfficer(true)}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                  isOfficer ? "bg-blue-500 text-white shadow-md" : "text-gray-600 hover:bg-gray-200"
                }`}
              >
                Officer Login
              </button>
            </div>
          </div>

          {/* Form */}
          <div className="space-y-4">
            <div className="relative">
              <FaEnvelope className="absolute top-1/2 left-4 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder={isOfficer ? "Email address" : "Email or Phone"}
                value={identifier}
                onChange={(e) => setIdentifier(e.target.value)}
                className="w-full pl-12 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-transparent"
              />
            </div>

            <div className="relative">
              <FaKey className="absolute top-1/2 left-4 transform -translate-y-1/2 text-gray-400" />
              <input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full pl-12 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-transparent"
              />
            </div>

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleLogin}
              disabled={isLoading}
              className={`w-full py-3 bg-gradient-to-r from-teal-500 to-blue-600 text-white rounded-lg text-lg font-semibold flex items-center justify-center gap-2 ${
                isLoading ? "opacity-70 cursor-not-allowed" : "hover:shadow-lg"
              }`}
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Authenticating...
                </>
              ) : (
                <>
                  <FaSignInAlt /> Login Now
                </>
              )}
            </motion.button>
          </div>

          {/* Signup Link */}
          <div className="mt-6 text-center">
            <p className="text-gray-600">
              Don't have an account?{' '}
              <Link 
                to="/register" 
                className="text-teal-600 font-medium hover:underline flex items-center justify-center"
              >
                <FaUserPlus className="mr-1" /> Sign up here
              </Link>
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="bg-gray-50 px-6 py-4 text-center">
          <p className="text-sm text-gray-500">
            By logging in, you agree to our{' '}
            <a href="#" className="text-blue-500 hover:underline">Terms of Service</a> and{' '}
            <a href="#" className="text-blue-500 hover:underline">Privacy Policy</a>
          </p>
        </div>
      </motion.div>
    </div>
  );
};

export default Login;