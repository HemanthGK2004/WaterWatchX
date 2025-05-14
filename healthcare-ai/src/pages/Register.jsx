import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { FaUser, FaPhone, FaEnvelope, FaMapMarkerAlt, FaIdCard, FaLock, FaKey } from "react-icons/fa";
import axios from "axios";

const Register = () => {
  const [formData, setFormData] = useState({
    name: "",
    phone: "+91",
    email: "",
    address: "",
    aadhar: "",
    password: "",
    otp: "",
  });
  const [otpSent, setOtpSent] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    if (name === "phone") {
      // Prevent modifying the "+91" prefix
      if (!value.startsWith("+91")) {
        return;
      }
      // Allow only digits after "+91" and limit to 10 digits
      const digits = value.slice(3).replace(/\D/g, "");
      if (digits.length <= 10) {
        setFormData({ ...formData, phone: "+91" + digits });
      }
    } else {
      setFormData({ ...formData, [name]: value });
    }
    setError(""); // Clear error on input change
  };

  const handleSendOtp = async () => {
    const phoneDigits = formData.phone.slice(3);
    if (!phoneDigits) {
      setError("Please enter a phone number");
      return;
    }
    if (!/^\d{10}$/.test(phoneDigits)) {
      setError("Please enter a valid 10-digit phone number");
      return;
    }
    setLoading(true);
    try {
      const response = await axios.post("http://localhost:5000/user/send_otp", {
        phone: formData.phone,
      });
      setSuccess(response.data.message);
      setOtpSent(true);
      setError("");
    } catch (err) {
      setError(err.response?.data?.error || "Failed to send OTP");
      setSuccess("");
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!otpSent) {
      setError("Please request and verify OTP first");
      return;
    }
    setLoading(true);
    try {
      const response = await axios.post("http://localhost:5000/user/register", formData);
      setSuccess(response.data.message);
      setError("");
      setTimeout(() => navigate("/login"), 2000); // Redirect to login after 2 seconds
    } catch (err) {
      setError(err.response?.data?.error || "Registration failed");
      setSuccess("");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#e0f2fe] to-[#bfdbfe] flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="max-w-md w-full bg-white rounded-2xl shadow-xl p-8 relative overflow-hidden"
      >
        {/* Wave Background */}
        <div className="absolute inset-0 opacity-10">
          <div className="wave"></div>
          <div className="wave wave2"></div>
        </div>

        <div className="relative">
          <h2 className="text-3xl font-bold text-center text-[#1e3a8a] mb-6">
            Join AquaAI-Report
          </h2>
          <p className="text-center text-[#4b5563] mb-8">
            Register to help solve water issues in your community!
          </p>

          {error && (
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-red-500 text-center mb-4"
            >
              {error}
            </motion.p>
          )}
          {success && (
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-green-500 text-center mb-4"
            >
              {success}
            </motion.p>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="flex items-center border border-[#bfdbfe] rounded-full px-4 py-2">
              <FaUser className="text-[#2563eb] mr-3" />
              <input
                type="text"
                name="name"
                value={formData.name}
                onChange={handleChange}
                placeholder="Full Name"
                className="w-full bg-transparent outline-none text-[#1e3a8a] placeholder-[#4b5563]"
                required
              />
            </div>

            <div className="flex items-center border border-[#bfdbfe] rounded-full px-4 py-2 relative">
              <FaPhone className="text-[#2563eb] mr-3" />
              <span className="absolute left-12 text-[#1e3a8a] font-semibold">IN</span>
              <input
                type="tel"
                name="phone"
                value={formData.phone}
                onChange={handleChange}
                placeholder="Enter 10-digit number"
                className="w-full bg-transparent outline-none text-[#1e3a8a] placeholder-[#4b5563] pl-12"
                required
              />
            </div>

            {!otpSent && (
              <motion.button
                type="button"
                onClick={handleSendOtp}
                disabled={loading}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="w-full bg-[#2563eb] text-white py-3 rounded-full font-semibold hover:bg-[#1e40af] disabled:opacity-50"
              >
                {loading ? "Sending OTP..." : "Send OTP"}
              </motion.button>
            )}

            {otpSent && (
              <div className="flex items-center border border-[#bfdbfe] rounded-full px-4 py-2">
                <FaKey className="text-[#2563eb] mr-3" />
                <input
                  type="text"
                  name="otp"
                  value={formData.otp}
                  onChange={handleChange}
                  placeholder="Enter OTP"
                  className="w-full bg-transparent outline-none text-[#1e3a8a] placeholder-[#4b5563]"
                  required
                />
              </div>
            )}

            <div className="flex items-center border border-[#bfdbfe] rounded-full px-4 py-2">
              <FaEnvelope className="text-[#2563eb] mr-3" />
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                placeholder="Email Address"
                className="w-full bg-transparent outline-none text-[#1e3a8a] placeholder-[#4b5563]"
                required
              />
            </div>

            <div className="flex items-center border border-[#bfdbfe] rounded-full px-4 py-2">
              <FaMapMarkerAlt className="text-[#2563eb] mr-3" />
              <input
                type="text"
                name="address"
                value={formData.address}
                onChange={handleChange}
                placeholder="Address"
                className="w-full bg-transparent outline-none text-[#1e3a8a] placeholder-[#4b5563]"
                required
              />
            </div>

            <div className="flex items-center border border-[#bfdbfe] rounded-full px-4 py-2">
              <FaIdCard className="text-[#2563eb] mr-3" />
              <input
                type="text"
                name="aadhar"
                value={formData.aadhar}
                onChange={handleChange}
                placeholder="Aadhar Number"
                className="w-full bg-transparent outline-none text-[#1e3a8a] placeholder-[#4b5563]"
                required
              />
            </div>

            <div className="flex items-center border border-[#bfdbfe] rounded-full px-4 py-2">
              <FaLock className="text-[#2563eb] mr-3" />
              <input
                type="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                placeholder="Password"
                className="w-full bg-transparent outline-none text-[#1e3a8a] placeholder-[#4b5563]"
                required
              />
            </div>

            <motion.button
              type="submit"
              disabled={loading || !otpSent}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="w-full bg-[#22c55e] text-white py-3 rounded-full font-semibold hover:bg-[#16a34a] disabled:opacity-50"
            >
              {loading ? "Registering..." : "Register"}
            </motion.button>
          </form>

          <p className="text-center text-[#4b5563] mt-6">
            Already have an account?{" "}
            <Link to="/login" className="text-[#2563eb] hover:underline">
              Log In
            </Link>
          </p>
        </div>
      </motion.div>

      {/* Inline CSS for Wave Animation */}
      <style>
        {`
          .wave {
            position: absolute;
            bottom: 0;
            width: 100%;
            height: 100px;
            background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 320'%3E%3Cpath fill='%23f0f9ff' fill-opacity='0.3' d='M0,128L48,138.7C96,149,192,171,288,181.3C384,192,480,192,576,181.3C672,171,768,149,864,138.7C960,128,1056,128,1152,144C1248,160,1344,192,1392,208L1440,224L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z'%3E%3C/path%3E%3C/svg%3E");
            animation: wave 8s linear infinite;
          }
          .wave2 {
            background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 320'%3E%3Cpath fill='%23f0f9ff' fill-opacity='0.5' d='M0,160L48,176C96,192,192,224,288,224C384,224,480,192,576,181.3C672,171,768,181,864,192C960,203,1056,213,1152,197.3C1248,181,1344,139,1392,117.3L1440,96L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z'%3E%3C/path%3E%3C/svg%3E");
            animation: wave 12s linear infinite reverse;
          }
          @keyframes wave {
            0% { transform: translateX(0); }
            100% { transform: translateX(-1440px); }
          }
        `}
      </style>
    </div>
  );
};

export default Register;