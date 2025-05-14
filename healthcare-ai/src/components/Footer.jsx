import React, { useState } from "react";
import { Link } from "react-router-dom";
import { FaWater, FaEnvelope, FaPhone, FaMapMarkerAlt, FaTwitter, FaFacebook, FaInstagram, FaGithub } from "react-icons/fa";
import { motion } from "framer-motion";

const Footer = () => {
  const [email, setEmail] = useState("");
  const [subscribed, setSubscribed] = useState(false);

  const handleSubscribe = (e) => {
    e.preventDefault();
    // Simulate newsletter subscription (replace with actual API call)
    if (email) {
      setSubscribed(true);
      setEmail("");
      setTimeout(() => setSubscribed(false), 3000);
    }
  };

  return (
    <footer className="relative bg-gradient-to-t from-[#1e3a8a] via-[#3b82f6] to-[#60a5fa] text-[#e6f0fa] py-16 px-8 overflow-hidden">
      {/* Wave Background Animation */}
      <div className="absolute inset-0 opacity-20">
        <div className="wave"></div>
        <div className="wave wave2"></div>
      </div>

      <div className="relative max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-12">
        {/* Brand Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="col-span-1"
        >
          <h1 className="text-3xl font-bold text-[#e0f2fe] flex items-center gap-2 mb-4">
            <FaWater className="text-[#bae6fd] animate-pulse" /> AquaAI-Report
          </h1>
          <p className="text-sm text-[#bfdbfe] mb-6">
            Harnessing AI to monitor and protect our water resources, ensuring sustainability and transparency for all.
          </p>
          <div className="flex space-x-4">
            {[
              { Icon: FaTwitter, link: "#" },
              { Icon: FaFacebook, link: "#" },
              { Icon: FaInstagram, link: "#" },
              { Icon: FaGithub, link: "#" },
            ].map(({ Icon, link }, index) => (
              <a
                key={index}
                href={link}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[#bae6fd] hover:text-[#f0f9ff] hover:scale-110 transition-all duration-300"
              >
                <Icon size={28} />
              </a>
            ))}
          </div>
        </motion.div>

        {/* Quick Links */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          <h6 className="font-semibold text-lg text-[#e0f2fe] mb-4">Water Insights</h6>
          <ul>
            {[
              { label: "Water Quality", to: "/water-quality" },
              { label: "AI Analysis", to: "/ai-analysis" },
              { label: "Reports", to: "/reports" },
              { label: "Data Upload", to: "/upload" },
            ].map((item) => (
              <li key={item.label} className="py-2">
                <Link
                  to={item.to}
                  className="text-sm hover:text-[#bae6fd] hover:pl-2 transition-all duration-300"
                >
                  {item.label}
                </Link>
              </li>
            ))}
          </ul>
        </motion.div>

        {/* Support Links */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          <h6 className="font-semibold text-lg text-[#e0f2fe] mb-4">Support</h6>
          <ul>
            {[
              { label: "Help Center", to: "/help" },
              { label: "FAQs", to: "/faqs" },
              { label: "Contact Us", to: "/contact" },
              { label: "Community Forum", to: "/forum" },
            ].map((item) => (
              <li key={item.label} className="py-2">
                <Link
                  to={item.to}
                  className="text-sm hover:text-[#bae6fd] hover:pl-2 transition-all duration-300"
                >
                  {item.label}
                </Link>
              </li>
            ))}
          </ul>
        </motion.div>

        {/* Newsletter Subscription */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
        >
          <h6 className="font-semibold text-lg text-[#e0f2fe] mb-4">Stay Updated</h6>
          <p className="text-sm text-[#bfdbfe] mb-4">
            Subscribe for the latest AI water reports and updates.
          </p>
          <form onSubmit={handleSubscribe} className="flex flex-col gap-3">
            <div className="relative">
              <FaEnvelope className="absolute top-1/2 left-3 transform -translate-y-1/2 text-[#bae6fd]" />
              <input
                type="email"
                placeholder="Enter your email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-[#1e40af] text-[#e6f0fa] rounded-full border border-[#60a5fa] focus:ring-2 focus:ring-[#bae6fd] focus:outline-none"
              />
            </div>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              type="submit"
              className="bg-[#2563eb] text-[#f0f9ff] py-2 rounded-full font-semibold hover:bg-[#1e40af] transition-all duration-300"
            >
              Subscribe
            </motion.button>
          </form>
          {subscribed && (
            <p className="text-sm text-[#bae6fd] mt-2">Thank you for subscribing!</p>
          )}
        </motion.div>
      </div>

      {/* Contact Info */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8, delay: 0.8 }}
        className="max-w-7xl mx-auto mt-12 border-t border-[#60a5fa] pt-8 flex flex-col md:flex-row justify-between items-center text-sm"
      >
        <div className="flex flex-col md:flex-row gap-4 text-[#bfdbfe] mb-4 md:mb-0">
          <span className="flex items-center gap-2">
            <FaPhone /> +1 (555) 123-4567
          </span>
          <span className="flex items-center gap-2">
            <FaEnvelope /> support@aquaai-report.com
          </span>
          <span className="flex items-center gap-2">
            <FaMapMarkerAlt /> 123 Ocean Ave, Blue City
          </span>
        </div>
        <p>© 2025 AquaAI-Report. All rights reserved.</p>
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
    </footer>
  );
};

export default Footer;