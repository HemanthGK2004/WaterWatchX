import axios from 'axios';
import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import { FaImage, FaMapMarkerAlt, FaNotesMedical, FaPercentage, FaComment } from 'react-icons/fa';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';

const OfficersDashboard = () => {
  const [reports, setReports] = useState([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [selectedReport, setSelectedReport] = useState(null);
  const [progress, setProgress] = useState(0);
  const [notes, setNotes] = useState('');
  const [progressImage, setProgressImage] = useState(null);
  const [resolvedImage, setResolvedImage] = useState(null);
  const [comments, setComments] = useState({}); // Comments keyed by report ID
  const [newComment, setNewComment] = useState(''); // Input for new comment

  const navigate = useNavigate();
  const API_URL = 'http://localhost:5000';
  const token = localStorage.getItem('token');

  // Fetch unresolved reports
  const fetchReports = async () => {
    try {
      const response = await axios.get(`${API_URL}/officer/reports`, {
        headers: { 'x-access-token': token },
      });
      if (Array.isArray(response.data)) {
        const unresolvedReports = response.data.filter(
          (report) => ['Pending', 'Accepted', 'In-Progress'].includes(report.status)
        );
        setReports(unresolvedReports);
      } else {
        console.error('Expected an array for reports:', response.data);
        setReports([]);
      }
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch reports');
      setLoading(false);
      if (err.response?.status === 401) {
        localStorage.removeItem('token');
        navigate('/login');
        toast.error('Session expired. Please log in again.', {
          position: 'top-right',
          autoClose: 3000,
        });
      } else {
        console.error('Error fetching reports:', err);
      }
    }
  };

  // Fetch comments for a specific report
  const fetchComments = async (reportId) => {
    try {
      const response = await axios.get(`${API_URL}/reports/${reportId}/comments`, {
        headers: { 'x-access-token': token },
      });
      setComments((prev) => ({
        ...prev,
        [reportId]: Array.isArray(response.data) ? response.data : [],
      }));
    } catch (err) {
      console.error(`Error fetching comments for report ${reportId}:`, err);
      toast.error('Failed to load comments', {
        position: 'top-right',
        autoClose: 3000,
      });
    }
  };

  // Post a new comment
  const postComment = async (reportId) => {
    if (!newComment.trim()) {
      toast.error('Comment cannot be empty', {
        position: 'top-right',
        autoClose: 3000,
      });
      return;
    }
    try {
      const response = await axios.post(
        `${API_URL}/reports/${reportId}/comments`,
        { content: newComment },
        { headers: { 'x-access-token': token } }
      );
      setComments((prev) => ({
        ...prev,
        [reportId]: [...(prev[reportId] || []), response.data],
      }));
      setNewComment('');
      toast.success('Comment posted successfully', {
        position: 'top-right',
        autoClose: 3000,
      });
    } catch (err) {
      const errorMessage = err.response?.data?.error || 'Failed to post comment';
      toast.error(errorMessage, {
        position: 'top-right',
        autoClose: 3000,
      });
      console.error('Error posting comment:', err);
    }
  };

  // Handle logout
  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/login');
    toast.info('Logged out successfully', {
      position: 'top-right',
      autoClose: 3000,
    });
  };

  useEffect(() => {
    if (!token) {
      navigate('/login');
      return;
    }
    fetchReports();
    const interval = setInterval(fetchReports, 30000); // Poll every 30 seconds
    return () => clearInterval(interval);
  }, [token, navigate]);

  // Handle report acceptance
  const handleAcceptReport = async (reportId) => {
    try {
      const response = await axios.put(
        `${API_URL}/accept_report`,
        { report_id: reportId },
        { headers: { 'x-access-token': token } }
      );
      setReports((prevReports) =>
        prevReports.map((report) =>
          report._id === reportId ? { ...report, status: 'Accepted' } : report
        )
      );
      const { notification } = response.data;
      const toastMessage = notification && notification.errors && notification.errors.length > 0
        ? `Report accepted. Notification issues: ${notification.errors.join(', ')}`
        : 'Report accepted and notification sent to user';
      toast[notification && notification.errors && notification.errors.length > 0 ? 'warn' : 'success'](toastMessage, {
        position: 'top-right',
        autoClose: 3000,
      });
      setError('');
    } catch (err) {
      if (err.response) {
        const errorMessage = err.response.status === 404
          ? 'Report not found or not assigned to you'
          : err.response.data?.error || 'Failed to accept report';
        setError(errorMessage);
        toast.error(errorMessage, {
          position: 'top-right',
          autoClose: 3000,
        });
      } else {
        const errorMessage = 'Network error or server is unreachable';
        setError(errorMessage);
        toast.error(errorMessage, {
          position: 'top-right',
          autoClose: 3000,
        });
      }
      console.error('Error accepting report:', err);
    }
  };

  // Handle starting progress
  const handleStartProgress = async (reportId) => {
    try {
      const formData = new FormData();
      formData.append('report_id', reportId);
      formData.append('progress', 0);
      formData.append('notes', 'Started working on the issue');
      const response = await axios.put(`${API_URL}/update_report_progress`, formData, {
        headers: { 'x-access-token': token },
      });
      setReports((prevReports) =>
        prevReports.map((report) =>
          report._id === reportId
            ? { ...report, status: 'In-Progress', progress: 0, progress_notes: 'Started working on the issue' }
            : report
        )
      );
      const { notification } = response.data;
      const toastMessage = notification.errors.length > 0
        ? `Progress started. Notification issues: ${notification.errors.join(', ')}`
        : 'Progress started and notification sent to user';
      toast[notification.errors.length > 0 ? 'warn' : 'success'](toastMessage, {
        position: 'top-right',
        autoClose: 3000,
      });
    } catch (err) {
      const errorMessage = err.response?.status === 404
        ? 'Report not found or not assigned to you'
        : 'Failed to start progress';
      setError(errorMessage);
      toast.error(errorMessage, {
        position: 'top-right',
        autoClose: 3000,
      });
      console.error('Error starting progress:', err);
    }
  };

  // Handle progress update
  const handleUpdateProgress = async () => {
    if (!selectedReport) return;
    if (progress < 0 || progress > 100) {
      toast.error('Progress must be between 0 and 100', {
        position: 'top-right',
        autoClose: 3000,
      });
      return;
    }
    try {
      const formData = new FormData();
      formData.append('report_id', selectedReport._id);
      formData.append('progress', progress);
      formData.append('notes', notes);
      if (progressImage) formData.append('progress_image', progressImage);
      const response = await axios.put(`${API_URL}/update_report_progress`, formData, {
        headers: { 'x-access-token': token },
      });
      setReports((prevReports) =>
        prevReports.map((report) =>
          report._id === selectedReport._id
            ? {
                ...report,
                status: 'In-Progress',
                progress,
                progress_notes: notes,
                progress_image: progressImage
                  ? `${API_URL}/uploads/${selectedReport._id}_progress.jpg`
                  : report.progress_image,
              }
            : report
        )
      );
      const notification = response.data.notification || { errors: [], sms_sent: false, email_sent: false };
      const toastMessage = notification.errors.length > 0
        ? `Progress updated. Notification issues: ${notification.errors.join(', ')}`
        : 'Progress updated and notification sent to user';
      toast[notification.errors.length > 0 ? 'warn' : 'success'](toastMessage, {
        position: 'top-right',
        autoClose: 3000,
      });
      setSelectedReport(null);
      setProgress(0);
      setNotes('');
      setProgressImage(null);
    } catch (err) {
      const errorMessage =
        err.response?.status === 404
          ? 'Report not found or not assigned to you'
          : err.response?.data?.error || 'Failed to update progress';
      setError(errorMessage);
      toast.error(errorMessage, {
        position: 'top-right',
        autoClose: 3000,
      });
      console.error('Error updating progress:', err);
    }
  };

  // Handle resolving report
  const handleResolveReport = async (reportId) => {
    try {
      const formData = new FormData();
      formData.append('report_id', reportId);
      if (resolvedImage) formData.append('resolved_image', resolvedImage);
      const response = await axios.put(`${API_URL}/update_report_status`, formData, {
        headers: { 'x-access-token': token },
      });
      setReports((prevReports) => prevReports.filter((report) => report._id !== reportId));
      setResolvedImage(null);
      fetchReports();
      const { notification } = response.data;
      const toastMessage = notification.errors.length > 0
        ? `Report resolved. Notification issues: ${notification.errors.join(', ')}`
        : 'Report resolved and notification sent to user';
      toast[notification.errors.length > 0 ? 'warn' : 'success'](toastMessage, {
        position: 'top-right',
        autoClose: 3000,
      });
    } catch (err) {
      const errorMessage = err.response?.status === 404
        ? 'Report not found or not assigned to you'
        : 'Failed to resolve report';
      setError(errorMessage);
      toast.error(errorMessage, {
        position: 'top-right',
        autoClose: 3000,
      });
      console.error('Error resolving report:', err);
    }
  };

  // Render action buttons based on report status
  const renderActions = (report) => {
    switch (report.status) {
      case 'Pending':
        return (
          <button
            className="bg-teal-600 text-white px-5 py-2 rounded-lg hover:bg-teal-700 transition duration-300 flex items-center space-x-2"
            onClick={() => handleAcceptReport(report._id)}
          >
            <FaNotesMedical />
            <span>Accept</span>
          </button>
        );
      case 'Accepted':
        return (
          <button
            className="bg-green-600 text-white px-5 py-2 rounded-lg hover:bg-green-700 transition duration-300 flex items-center space-x-2"
            onClick={() => handleStartProgress(report._id)}
          >
            <FaNotesMedical />
            <span>Start Progress</span>
          </button>
        );
      case 'In-Progress':
        return (
          <div className="flex space-x-3">
            <button
              className="bg-yellow-600 text-white px-5 py-2 rounded-lg hover:bg-yellow-700 transition duration-300 flex items-center space-x-2"
              onClick={() => setSelectedReport(report)}
            >
              <FaNotesMedical />
              <span>Update Progress</span>
            </button>
            <button
              className="bg-red-600 text-white px-5 py-2 rounded-lg hover:bg-red-700 transition duration-300 flex items-center space-x-2"
              onClick={() => document.getElementById(`resolve-${report._id}`).click()}
            >
              <FaNotesMedical />
              <span>Resolve</span>
            </button>
            <input
              id={`resolve-${report._id}`}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                setResolvedImage(e.target.files[0]);
                handleResolveReport(report._id);
              }}
            />
          </div>
        );
      default:
        return null;
    }
  };

  // Status badge styling
  const getStatusBadgeClass = (status) => {
    switch (status) {
      case 'Pending':
        return 'bg-teal-100 text-teal-800';
      case 'Accepted':
        return 'bg-green-100 text-green-800';
      case 'In-Progress':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-100 via-blue-200 to-purple-200 p-6 overflow-hidden">
      <motion.header
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, type: 'spring' }}
        className="text-center mb-8 flex justify-between items-center"
      >
        <div>
          <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-teal-500 to-purple-600">
            Unresolved Reports
          </h1>
          <p className="text-gray-700 mt-2 italic">“Working Together to Restore Our Waters!”</p>
        </div>
      </motion.header>

      <div className="max-w-7xl mx-auto">
        {error && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center p-4 bg-red-100 text-red-600 rounded-lg mb-6 font-medium"
          >
            {error}
          </motion.div>
        )}
        {loading ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center p-8 bg-white/80 backdrop-blur-md rounded-xl shadow-lg"
          >
            <FaImage className="text-5xl text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600 text-lg animate-pulse">Loading reports...</p>
          </motion.div>
        ) : reports.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center p-8 bg-white/80 backdrop-blur-md rounded-xl shadow-lg"
          >
            <FaImage className="text-5xl text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600 text-lg">No unresolved reports yet.</p>
          </motion.div>
        ) : (
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {reports.map((report, index) => (
              <motion.div
                key={report._id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="p-6 bg-white/90 backdrop-blur-md rounded-xl shadow-lg border border-teal-100 hover:shadow-xl transition-shadow w-full"
                onMouseEnter={() => fetchComments(report._id)} // Fetch comments on hover
              >
                <div className="flex items-center justify-between mb-3 flex-wrap">
                  <h2 className="text-xl font-semibold text-teal-700 flex items-center space-x-2 break-words max-w-full">
                    <FaMapMarkerAlt className="text-teal-500" />
                    <span>{report.address}</span>
                  </h2>
                  <span
                    className={`text-sm font-semibold px-3 py-1 rounded-full ${getStatusBadgeClass(
                      report.status
                    )}`}
                  >
                    {report.status}
                  </span>
                </div>
                <div className="space-y-3 text-gray-700">
                  <p className="flex items-center space-x-2">
                    <FaNotesMedical className="text-gray-500" />
                    <span>
                      <span className="font-medium">Issue:</span>{' '}
                      <span
                        className={
                          report.status === 'Pending'
                            ? 'text-blue-500'
                            : report.status === 'Accepted'
                            ? 'text-green-500'
                            : 'text-yellow-500'
                        }
                      >
                        {report.status}
                      </span>
                    </span>
                  </p>
                  <p className="flex items-center space-x-2">
                    <FaPercentage className="text-gray-500" />
                    <span>
                      <span className="font-medium">Confidence:</span>{' '}
                      {(report.confidence * 100).toFixed(2)}%
                    </span>
                  </p>
                  {report.status === 'In-Progress' && (
                    <>
                      <p className="flex items-center space-x-2">
                        <FaPercentage className="text-teal-500" />
                        <span>
                          <span className="font-medium">Progress:</span> {report.progress}%
                        </span>
                      </p>
                      <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                        <div
                          className="bg-gradient-to-r from-teal-500 to-purple-600 h-3 rounded-full transition-all duration-500"
                          style={{ width: `${report.progress}%` }}
                        ></div>
                      </div>
                      {report.progress_notes && (
                        <p className="flex items-center space-x-2">
                          <FaNotesMedical className="text-gray-500" />
                          <span>
                            <span className="font-medium">Notes:</span> {report.progress_notes}
                          </span>
                        </p>
                      )}
                    </>
                  )}
                  <div>
                    <span className="font-medium">Image:</span>{' '}
                    {report.image ? (
                      <div className="mt-2">
                        <img
                          src={report.image}
                          alt="Report"
                          className="w-full h-64 object-cover rounded-md"
                          onError={(e) => {
                            console.error('Failed to load report image:', report.image);
                            e.target.src = 'https://via.placeholder.com/150?text=Image+Not+Found';
                          }}
                          onLoad={() => console.log('Report image loaded successfully:', report.image)}
                        />
                        <a
                          href={report.image}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-teal-500 hover:underline text-sm mt-1 inline-block"
                        >
                          Open in new tab
                        </a>
                      </div>
                    ) : (
                      <span>No Image</span>
                    )}
                  </div>
                  {report.status === 'In-Progress' && (
                    <div>
                      <span className="font-medium">Progress Image:</span>{' '}
                      {report.progress_image ? (
                        <div className="mt-2">
                          <img
                            src={report.progress_image}
                            alt="Progress"
                            className="w-full h-64 object-cover rounded-md"
                            onError={(e) => {
                              console.error('Failed to load progress image:', report.progress_image);
                              e.target.src = 'https://via.placeholder.com/150?text=Image+Not+Found';
                            }}
                            onLoad={() =>
                              console.log('Progress image loaded successfully:', report.progress_image)
                            }
                          />
                          <a
                            href={report.progress_image}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-teal-500 hover:underline text-sm mt-1 inline-block"
                          >
                            Open in new tab
                          </a>
                        </div>
                      ) : (
                        <span>No Progress Image</span>
                      )}
                    </div>
                  )}
                  {/* Comment Section */}
                  <div className="mt-4">
                    <h3 className="text-lg font-semibold text-teal-700 flex items-center space-x-2">
                      <FaComment className="text-teal-500" />
                      <span>Comments</span>
                    </h3>
                    <div className="mt-2 max-h-40 overflow-y-auto">
                      {(comments[report._id] || []).length > 0 ? (
                        comments[report._id].map((comment, idx) => (
                          <div
                            key={idx}
                            className="p-3 bg-teal-50 rounded-lg mb-2 border border-teal-200"
                          >
                            <p className="text-gray-700">{comment.content}</p>
                            <p className="text-sm text-gray-500 mt-1">
                              By {comment.author || 'Officer'} on{' '}
                              {new Date(comment.createdAt).toLocaleString()}
                            </p>
                          </div>
                        ))
                      ) : (
                        <p className="text-gray-600">No comments yet.</p>
                      )}
                    </div>
                    <div className="mt-3 flex items-center space-x-2">
                      <input
                        type="text"
                        value={newComment}
                        onChange={(e) => setNewComment(e.target.value)}
                        placeholder="Add a comment..."
                        className="flex-1 border border-gray-300 rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-teal-500"
                      />
                      <button
                        className="bg-teal-600 text-white px-4 py-2 rounded-lg hover:bg-teal-700 transition duration-200"
                        onClick={() => postComment(report._id)}
                      >
                        Post
                      </button>
                    </div>
                  </div>
                </div>
                <div className="mt-6 flex justify-end">{renderActions(report)}</div>
              </motion.div>
            ))}
          </div>
        )}
      </div>

      <motion.footer
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="mt-8 text-center text-gray-600"
      >
        <p className="font-semibold">“Together, We’re Making a Difference!”</p>
      </motion.footer>

      {/* Progress Update Modal */}
      {selectedReport && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
        >
          <div className="bg-white/95 backdrop-blur-md p-8 rounded-xl shadow-2xl w-full max-w-lg">
            <h2 className="text-2xl font-bold text-teal-700 mb-6 flex items-center space-x-2">
              <FaNotesMedical className="text-teal-500" />
              <span>Update Progress for {selectedReport.address}</span>
            </h2>
            <div className="mb-6">
              <label className="block text-gray-700 font-medium mb-2">Progress (%)</label>
              <input
                type="range"
                min="0"
                max="100"
                value={progress}
                onChange={(e) => setProgress(Number(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-teal-600"
              />
              <span className="block text-gray-600 mt-2">{progress}%</span>
            </div>
            <div className="mb-6">
              <label className="block text-gray-700 font-medium mb-2">Notes</label>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                className="w-full border border-gray-300 rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-teal-500 transition duration-200"
                rows="4"
                placeholder="Enter progress notes..."
              ></textarea>
            </div>
            <div className="mb-6">
              <label className="block text-gray-700 font-medium mb-2">Progress Image</label>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => setProgressImage(e.target.files[0])}
                className="w-full border border-gray-300 rounded-lg p-2"
              />
            </div>
            <div className="flex justify-end space-x-4">
              <button
                className="bg-gray-500 text-white px-5 py-2 rounded-lg hover:bg-gray-600 transition duration-200 flex items-center space-x-2"
                onClick={() => {
                  setSelectedReport(null);
                  setProgress(0);
                  setNotes('');
                  setProgressImage(null);
                }}
              >
                <FaNotesMedical />
                <span>Cancel</span>
              </button>
              <button
                className="bg-teal-600 text-white px-5 py-2 rounded-lg hover:bg-teal-700 transition duration-200 flex items-center space-x-2"
                onClick={handleUpdateProgress}
              >
                <FaNotesMedical />
                <span>Update</span>
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default OfficersDashboard;