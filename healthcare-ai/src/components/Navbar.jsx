import React, { useEffect, useState } from "react";
import { AiOutlineClose, AiOutlineMenu } from "react-icons/ai";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const Navbar = () => {
  const [nav, setNav] = useState(false);
  const { isLoggedIn, role, logout } = useAuth();
  const navigate = useNavigate();

  const handleNav = () => {
    console.log("handleNav called, nav state:", !nav);
    setNav(!nav);
  };

  const handleLogout = () => {
    console.log("handleLogout triggered");
    logout();
    navigate("/login");
  };

  const publicRoutes = [
    { to: "/dashboard", label: "Home" },
    { to: "/initiatives", label: "Water Initiatives" },
    { to: "/community", label: "Community Dashboard" },
    { to: "/contact", label: "Contact" },
    { to: "/reports", label: "Reports" },
    { to: "/ai-analysis", label: "AI Analysis" },
  ];

  const userRoutes = [
    { to: "/UserDashboard", label: "Dashboard" },
    { to: "/water_quality", label: "Water Quality" },
    { to: "/upload", label: "Data Upload" },
    { to: "/user-reports", label: "My Reports" },
    { to: "/community", label: "Community Dashboard" },
  ];

  const officerRoutes = [
    { to: "/officer/profile", label: "Officer Dashboard" },
    { to: "/officer/reports", label: "Assigned Reports" },
    { to: "/officer/resolved-reports", label: "Resolved Reports" },
  ];

  const displayedRoutes = isLoggedIn
    ? role === "officer"
      ? officerRoutes
      : userRoutes
    : publicRoutes;

  useEffect(() => {
    console.log("Navbar re-rendered, isLoggedIn:", isLoggedIn, "role:", role, "Routes:", displayedRoutes);
  }, [isLoggedIn, role, displayedRoutes]);

  return (
    <div className="flex items-center h-24 w-full px-8 text-[#e6f0fa] bg-gradient-to-r from-[#1e3a8a] via-[#3b82f6] to-[#60a5fa] shadow-xl border-b-4 border-[#93c5fd]">
      <h1 className="text-4xl font-bold text-[#e0f2fe] drop-shadow-lg tracking-tight">
        AquaAI-Report 💧
      </h1>

      <div className="hidden md:flex items-center flex-1 justify-center">
        <ul className="flex items-center space-x-12">
          {displayedRoutes.map((item) => (
            <li
              key={item.to}
              className="p-4 relative group cursor-pointer text-lg font-medium text-[#e6f0fa] transition-all duration-300"
            >
              <Link
                to={item.to}
                onClick={() => console.log("Navigating to:", item.to)}
              >
                {item.label}
                <span className="absolute left-0 bottom-0 w-0 h-1 bg-[#bae6fd] rounded-full group-hover:w-full transition-all duration-500 ease-out"></span>
              </Link>
            </li>
          ))}
        </ul>

        {isLoggedIn ? (
          <button
            onClick={handleLogout}
            className="ml-8 bg-red-500 px-8 py-2 rounded-full text-white font-semibold hover:bg-red-600 transition-all duration-300"
          >
            Logout
          </button>
        ) : (
          <Link to="/login" className="ml-8">
            <button className="relative bg-[#2563eb] px-8 py-2 rounded-full text-[#f0f9ff] font-semibold transition-all duration-300 hover:bg-[#1e40af] hover:scale-105 shadow-md hover:shadow-lg overflow-hidden group">
              <span className="relative z-10">Login</span>
              <span className="absolute inset-0 bg-gradient-to-r from-[#60a5fa] to-transparent opacity-0 group-hover:opacity-30 rounded-full transition-opacity duration-300"></span>
            </button>
          </Link>
        )}
      </div>

      <div onClick={handleNav} className="block md:hidden cursor-pointer z-20">
        {nav ? (
          <AiOutlineClose
            size={28}
            className="text-[#e0f2fe] transition-transform duration-300 rotate-180"
          />
        ) : (
          <AiOutlineMenu
            size={28}
            className="text-[#e0f2fe] transition-transform duration-300"
          />
        )}
      </div>

      <ul
        className={`fixed top-0 left-0 w-[80%] h-full bg-gradient-to-b from-[#1e3a8a] to-[#3b82f6] border-r-4 border-[#93c5fd] transform transition-all duration-500 ease-in-out z-10 ${
          nav ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <h1 className="text-4xl font-bold text-[#e0f2fe] m-8 drop-shadow-lg">
          AquaAI-Report 💧
        </h1>
        {displayedRoutes.map((item) => (
          <li
            key={item.to}
            className="p-5 border-b border-[#93c5fd] cursor-pointer hover:bg-[#2563eb] transition-colors duration-300"
          >
            <Link
              to={item.to}
              onClick={() => {
                console.log("Mobile nav to:", item.to);
                handleNav();
              }}
              className="text-lg font-medium text-[#e6f0fa] hover:text-[#f0f9ff]"
            >
              {item.label}
            </Link>
          </li>
        ))}

        {isLoggedIn ? (
          <li className="p-5">
            <button
              onClick={() => {
                handleLogout();
                handleNav();
              }}
              className="w-full bg-red-500 py-3 rounded-full text-white font-semibold transition-all duration-300 hover:bg-red-600"
            >
              Logout
            </button>
          </li>
        ) : (
          <li className="p-5">
            <Link to="/login" onClick={handleNav}>
              <button className="w-full bg-[#2563eb] py-3 rounded-full text-[#f0f9ff] font-semibold transition-all duration-300 hover:bg-[#1e40af] hover:shadow-lg">
                Login
              </button>
            </Link>
          </li>
        )}
      </ul>

      {nav && (
        <div
          className="fixed inset-0 bg-black bg-opacity-60 z-0 md:hidden"
          onClick={handleNav}
        ></div>
      )}
    </div>
  );
};

export default Navbar;