import React from "react";
import { Navigate, Route, BrowserRouter as Router, Routes } from "react-router-dom";
import Chatbot from "./components/Chatbot";
import CommunityDashboard from "./components/CommunityDashboard";
import FlowDashboard from "./components/FlowDashboard";
import Footer from "./components/Footer";
import Navbar from "./components/Navbar";
import ProtectedRoute from "./components/ProtectedRoute";
import WaterQualityForm from "./components/WaterQualityForm";
import { AuthProvider } from "./context/AuthContext";

// Importing Pages
import Contact from "./pages/Contact";
import Dashboard from "./pages/Dashboard";
import FlowOptimization from "./pages/FlowOptimization";
import HealthInitiatives from "./pages/HealthInitiatives";
import Login from "./pages/Login";
import OfficerProfile from "./pages/OfficerProfile";
import OfficersDashboard from "./pages/OfficersDashboard";
import Register from "./pages/Register";
import Reports from "./pages/Reports";
import ResolvedReports from "./pages/ResolvedReports";
import UploadPage from "./pages/UploadPage";
import UserDashboard from "./pages/UserDashboard";
import UserReports from "./pages/UserReports";
import WaterIssueMap from "./pages/WaterIssueMap";

function App() {
  const token = localStorage.getItem('token');

  return (
    <AuthProvider>
      <Router>
        <Navbar />
        <Routes>
          {/* Public Routes */}
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/register" element={<Register />} />
          <Route path="/ai-analysis" element={<WaterIssueMap />} />
          <Route path="/initiatives" element={<HealthInitiatives />} />
          <Route path="/community" element={<CommunityDashboard />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="/login" element={<Login />} />

          {/* Protected Routes for Users */}
          <Route
            path="/UserDashboard"
            element={
              <ProtectedRoute requiredRole="user">
                <UserDashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/user-reports"
            element={
              <ProtectedRoute requiredRole="user">
                <UserReports />
              </ProtectedRoute>
            }
          />
          <Route
            path="/water_quality"
            element={
              <ProtectedRoute requiredRole="user">
                <WaterQualityForm token={token} />
              </ProtectedRoute>
            }
          />
          <Route
            path="/upload"
            element={
              <ProtectedRoute requiredRole="user">
                <UploadPage />
              </ProtectedRoute>
            }
          />

          {/* Protected Routes for Officers */}
          <Route
            path="/officers-dashboard"
            element={
              <ProtectedRoute requiredRole="officer">
                <OfficersDashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/officer/profile"
            element={
              <ProtectedRoute requiredRole="officer">
                <OfficerProfile />
              </ProtectedRoute>
            }
          />
          <Route
            path="/officer/reports"
            element={
              <ProtectedRoute requiredRole="officer">
                <OfficersDashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/officer/resolved-reports"
            element={
              <ProtectedRoute requiredRole="officer">
                <ResolvedReports />
              </ProtectedRoute>
            }
          />
          <Route
            path="/flow-dashboard"
            element={
              <ProtectedRoute requiredRole="officer">
                <FlowDashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/officer/flow-optimization"
            element={
              <ProtectedRoute requiredRole="officer">
                <FlowOptimization />
              </ProtectedRoute>
            }
          />

          {/* Fallback Route */}
          <Route path="*" element={<Navigate to="/dashboard" />} />
        </Routes>
        <Chatbot />
        <Footer />
      </Router>
    </AuthProvider>
  );
}

export default App;