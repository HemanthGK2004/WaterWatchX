import React from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const ProtectedRoute = ({ children }) => {
  const { isLoggedIn } = useAuth();
  console.log("ProtectedRoute - isLoggedIn:", isLoggedIn);
  if (isLoggedIn === null) {
    return <div>Loading...</div>;
  }
  return isLoggedIn ? children : <Navigate to="/login" />;
};

export default ProtectedRoute;