import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion, useAnimation } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { Droplet, BarChart3, MapPin, Users, CheckCircle, Map, ArrowRight, Leaf, CloudRain, Waves } from "lucide-react";

const Dashboard = () => {
  const [reportStats, setReportStats] = useState({ total: 0, resolved: 0 });
  const controls = useAnimation();
  const [ref, inView] = useInView({ triggerOnce: true });
  const [currentQuote, setCurrentQuote] = useState(0);

  const quotes = [
    {
      text: "Water is the driving force of all nature. Join us in protecting it.",
      author: "Leonardo da Vinci"
    },
    {
      text: "Thousands have lived without love, not one without water.",
      author: "W.H. Auden"
    },
    {
      text: "A drop of water is worth more than a sack of gold to a thirsty man.",
      author: "Proverb"
    }
  ];

  // Fetch report statistics from backend
  useEffect(() => {
    fetch("http://localhost:5000/get_reports")
      .then((response) => response.json())
      .then((data) => {
        const total = data.length;
        const resolved = data.filter((report) => report.resolved).length;
        setReportStats({ total, resolved });
      })
      .catch((error) => console.error("Error fetching reports:", error));

    // Rotate quotes every 8 seconds
    const quoteInterval = setInterval(() => {
      setCurrentQuote((prev) => (prev + 1) % quotes.length);
    }, 8000);

    return () => clearInterval(quoteInterval);
  }, []);

  // Animate when section comes into view
  useEffect(() => {
    if (inView) {
      controls.start("visible");
    }
  }, [controls, inView]);

  const counterVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 1 } },
  };

  const features = [
    {
      icon: <Droplet className="h-10 w-10 text-blue-600" />,
      title: "Report Water Issues",
      description: "Easily report leaks, contamination, or other water issues in your community.",
      link: "/upload",
      color: "bg-blue-100",
    },
    {
      icon: <BarChart3 className="h-10 w-10 text-teal-600" />,
      title: "Track Community Data",
      description: "View real-time analytics on water issues through our interactive dashboard.",
      link: "/reports",
      color: "bg-teal-100",
    },
    {
      icon: <MapPin className="h-10 w-10 text-indigo-600" />,
      title: "Visualize Issues on Map",
      description: "Explore reported issues on an interactive map to identify problem areas.",
      link: "/ai-analysis",
      color: "bg-indigo-100",
    },
  ];

  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-b from-blue-50 to-teal-50">
      {/* Hero Section */}
      <section className="relative bg-gradient-to-br from-blue-900 via-blue-800 to-teal-800 text-white overflow-hidden">
        <div className="absolute inset-0 opacity-20">
          <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-30"></div>
        </div>
        <div className="container mx-auto px-4 py-24 relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="max-w-3xl mx-auto text-center"
          >
            <div className="flex justify-center mb-4">
              <Droplet className="h-12 w-12 text-blue-300 animate-pulse" />
            </div>
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6 leading-tight">
              AquaAI-Report
              <span className="block text-blue-300 bg-gradient-to-r from-blue-300 to-teal-300 bg-clip-text text-transparent">
                Protecting Our Water Future
              </span>
            </h1>
            
            <motion.div
              key={currentQuote}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8 }}
              className="mb-8"
            >
              <p className="text-xl md:text-2xl text-blue-100 italic mb-2">
                "{quotes[currentQuote].text}"
              </p>
              <p className="text-blue-200">— {quotes[currentQuote].author}</p>
            </motion.div>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <motion.div
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.98 }}
              >
                <Link
                  to="/upload"
                  className="inline-flex items-center justify-center bg-blue-500 hover:bg-blue-600 text-white py-3 px-8 rounded-full font-medium transition-all duration-300 shadow-lg hover:shadow-xl"
                >
                  Report an Issue <ArrowRight className="ml-2" />
                </Link>
              </motion.div>
              <motion.div
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.98 }}
              >
                <Link
                  to="/reports"
                  className="inline-flex items-center justify-center bg-transparent border-2 border-blue-300 hover:border-white text-blue-100 hover:text-white py-3 px-8 rounded-full font-medium transition-all duration-300"
                >
                  View Dashboard
                </Link>
              </motion.div>
            </div>
          </motion.div>
        </div>
        <div className="absolute bottom-0 left-0 right-0">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 200">
            <path
              fill="#f9fafb"
              fillOpacity="1"
              d="M0,128L80,138.7C160,149,320,171,480,165.3C640,160,800,128,960,117.3C1120,107,1280,117,1360,122.7L1440,128L1440,320L1360,320C1280,320,1120,320,960,320C800,320,640,320,480,320C320,320,160,320,80,320L0,320Z"
            ></path>
          </svg>
        </div>
      </section>

      {/* Join Us Section */}
      <section className="py-16 bg-white">
        <div className="container mx-auto px-4 text-center">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
            className="text-3xl font-bold mb-4 text-gray-800"
          >
            Be Part of the Solution 💧
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            viewport={{ once: true }}
            className="text-xl text-gray-600 max-w-3xl mx-auto mb-8"
          >
            "We never know the worth of water till the well is dry." - Thomas Fuller
          </motion.p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <motion.div whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.98 }}>
              <Link to="/register">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="bg-gradient-to-r from-teal-500 to-green-500 px-8 py-4 rounded-full text-white text-lg font-semibold hover:from-teal-600 hover:to-green-600 shadow-lg"
                >
                  Join Our Community
                </motion.button>
              </Link>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 bg-gray-50">
        <div className="container mx-auto px-4">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
            className="text-3xl font-bold mb-12 text-center text-gray-800"
          >
            How You Can Make a Difference
          </motion.h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                whileHover={{ y: -5 }}
                className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all"
              >
                <div className={`${feature.color} w-20 h-20 rounded-full flex items-center justify-center mb-6 mx-auto`}>
                  {feature.icon}
                </div>
                <h3 className="text-xl font-bold text-center text-gray-800 mb-3">
                  {feature.title}
                </h3>
                <p className="text-gray-600 text-center mb-6">
                  {feature.description}
                </p>
                <div className="text-center">
                  <Link
                    to={feature.link}
                    className="inline-flex items-center text-blue-600 hover:text-blue-800 font-medium"
                  >
                    Learn more <ArrowRight className="ml-1 h-4 w-4" />
                  </Link>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Community Impact Section */}
      <section className="py-16 bg-white" ref={ref}>
        <div className="container mx-auto px-4">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
            className="text-3xl font-bold mb-12 text-center text-gray-800"
          >
            Our Collective Impact
          </motion.h2>
          <motion.div
            variants={counterVariants}
            initial="hidden"
            animate={controls}
            className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto"
          >
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all">
              <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mb-4 mx-auto">
                <CheckCircle className="h-8 w-8 text-blue-600" />
              </div>
              <p className="text-5xl font-bold text-blue-600 text-center">
                {reportStats.resolved}+
              </p>
              <h3 className="text-xl font-semibold text-gray-800 mt-2 text-center">
                Issues Resolved
              </h3>
              <p className="text-gray-600 mt-4 text-center">
                "Many small drops make a mighty ocean." Together we've restored clean water access.
              </p>
            </div>
            <div className="bg-gradient-to-br from-teal-50 to-teal-100 rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all">
              <div className="bg-teal-100 w-16 h-16 rounded-full flex items-center justify-center mb-4 mx-auto">
                <Droplet className="h-8 w-8 text-teal-600" />
              </div>
              <p className="text-5xl font-bold text-teal-600 text-center">
                {reportStats.total}+
              </p>
              <h3 className="text-xl font-semibold text-gray-800 mt-2 text-center">
                Reports Submitted
              </h3>
              <p className="text-gray-600 mt-4 text-center">
                "Water is life's matter and matrix, mother and medium." Your reports make a difference.
              </p>
            </div>
            <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all">
              <div className="bg-indigo-100 w-16 h-16 rounded-full flex items-center justify-center mb-4 mx-auto">
                <Users className="h-8 w-8 text-indigo-600" />
              </div>
              <p className="text-5xl font-bold text-indigo-600 text-center">
                {Math.round((reportStats.resolved || 1000) / 1000)}K+
              </p>
              <h3 className="text-xl font-semibold text-gray-800 mt-2 text-center">
                Lives Impacted
              </h3>
              <p className="text-gray-600 mt-4 text-center">
                "Water connects us all." Your actions create ripples of positive change.
              </p>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Water Conservation Tips */}
      <section className="py-16 bg-gradient-to-br from-blue-50 to-teal-50">
        <div className="container mx-auto px-4">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
            className="text-3xl font-bold mb-12 text-center text-gray-800"
          >
            Water Conservation Tips
          </motion.h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
            {[
              {
                icon: <Leaf className="h-8 w-8 text-green-600" />,
                title: "Fix Leaks",
                tip: "A dripping faucet can waste 20 gallons of water per day."
              },
              {
                icon: <CloudRain className="h-8 w-8 text-blue-600" />,
                title: "Collect Rainwater",
                tip: "Use rain barrels to water plants and gardens."
              },
              {
                icon: <Droplet className="h-8 w-8 text-teal-600" />,
                title: "Shorter Showers",
                tip: "Cutting shower time by 2 minutes saves 5 gallons."
              },
              {
                icon: <Waves className="h-8 w-8 text-indigo-600" />,
                title: "Full Loads Only",
                tip: "Run dishwashers and washing machines only when full."
              }
            ].map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                whileHover={{ scale: 1.03 }}
                className="bg-white rounded-xl p-6 shadow-md hover:shadow-lg transition-all flex flex-col items-center text-center"
              >
                <div className="bg-blue-50 w-16 h-16 rounded-full flex items-center justify-center mb-4">
                  {item.icon}
                </div>
                <h3 className="text-xl font-semibold text-gray-800 mb-2">
                  {item.title}
                </h3>
                <p className="text-gray-600">{item.tip}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="py-16 bg-gradient-to-br from-blue-900 to-teal-800 text-white">
        <div className="container mx-auto px-4 text-center">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
            className="text-3xl font-bold mb-6"
          >
            Ready to Make a Difference?
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            viewport={{ once: true }}
            className="text-xl mb-8 max-w-2xl mx-auto"
          >
            "Water is the most critical resource issue of our lifetime and our children's lifetime." - Lyndon B. Johnson
          </motion.p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <motion.div
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.98 }}
            >
              <Link
                to="/register"
                className="inline-flex items-center justify-center bg-white text-blue-600 hover:bg-blue-50 py-4 px-8 rounded-full text-lg font-semibold shadow-lg hover:shadow-xl transition-all"
              >
                Join Now
              </Link>
            </motion.div>
            <motion.div
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.98 }}
            >
              <Link
                to="/about"
                className="inline-flex items-center justify-center bg-transparent border-2 border-white text-white hover:bg-white/10 py-4 px-8 rounded-full text-lg font-semibold transition-all"
              >
                Learn More
              </Link>
            </motion.div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Dashboard;