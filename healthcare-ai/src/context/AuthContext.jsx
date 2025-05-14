import React, { createContext, useState, useEffect, useContext } from "react";

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [isLoggedIn, setIsLoggedIn] = useState(null);
  const [role, setRole] = useState(null);

  useEffect(() => {
    const validateToken = async () => {
      const token = localStorage.getItem("token");
      const storedRole = localStorage.getItem("role");
      console.log("Initial token check:", token, "role:", storedRole);
      if (token) {
        setIsLoggedIn(true);
        setRole(storedRole || null);
      } else {
        setIsLoggedIn(false);
        setRole(null);
      }
    };
    validateToken();
  }, []);

  const login = (token, userRole) => {
    console.log("Login called with token:", token, "role:", userRole);
    localStorage.setItem("token", token);
    localStorage.setItem("role", userRole);
    setIsLoggedIn(true);
    setRole(userRole);
  };

  const logout = () => {
    console.log("Logout called");
    localStorage.removeItem("token");
    localStorage.removeItem("role");
    setIsLoggedIn(false);
    setRole(null);
  };

  useEffect(() => {
    const handleStorageChange = () => {
      const token = localStorage.getItem("token");
      const storedRole = localStorage.getItem("role");
      console.log("Storage event triggered, token:", token, "role:", storedRole);
      setIsLoggedIn(!!token);
      setRole(storedRole || null);
    };
    window.addEventListener("storage", handleStorageChange);
    return () => window.removeEventListener("storage", handleStorageChange);
  }, []);

  if (isLoggedIn === null) {
    return <div>Loading...</div>;
  }

  return (
    <AuthContext.Provider value={{ isLoggedIn, role, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};