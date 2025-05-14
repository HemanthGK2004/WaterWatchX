import axios from 'axios';
import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import CommunityLeaderboard from './CommunityLeaderboard';

const CommunityDashboard = () => {
  const { isLoggedIn } = useAuth();
  const [reports, setReports] = useState([]);
  const [comments, setComments] = useState({});
  const [newComment, setNewComment] = useState({});
  const [error, setError] = useState(null);
  const [commentLoading, setCommentLoading] = useState(false);
  const token = localStorage.getItem('token');

  const fetchReportsAndComments = async () => {
    setError(null); // Clear any previous error before fetching
    try {
      const reportsResponse = await axios.get('http://localhost:5000/community_reports');
      setReports(reportsResponse.data);
      // Fetch comments for each report
      const commentsPromises = reportsResponse.data.map((report) =>
        axios.get(`http://localhost:5000/get_comments?report_id=${report._id}`)
      );
      const commentsResponses = await Promise.all(commentsPromises);
      const commentsData = {};
      reportsResponse.data.forEach((report, index) => {
        commentsData[report._id] = commentsResponses[index].data;
      });
      setComments(commentsData);
      console.log('Fetched reports:', reportsResponse.data); // Debugging log
      console.log('Fetched comments:', commentsData); // Debugging log
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to fetch community reports and comments');
    }
  };

  const handleUpvote = async (reportId) => {
    if (!isLoggedIn || !token) {
      setError('Please log in to upvote reports');
      return;
    }
    setError(null); // Clear previous errors before upvoting
    try {
      await axios.post(
        'http://localhost:5000/upvote_report',
        { report_id: reportId },
        { headers: { 'x-access-token': token } }
      );
      fetchReportsAndComments(); // Refresh reports after upvoting
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to upvote report');
    }
  };

  const handleCommentSubmit = async (reportId) => {
    if (!isLoggedIn || !token) {
      setError('Please log in to post a comment');
      return;
    }
    if (!newComment[reportId]?.trim()) {
      setError('Comment cannot be empty');
      return;
    }
    setError(null); // Clear previous errors before commenting
    setCommentLoading(true);
    try {
      await axios.post(
        'http://localhost:5000/add_comment',
        { report_id: reportId, comment: newComment[reportId] },
        { headers: { 'x-access-token': token } }
      );
      const commentsResponse = await axios.get(`http://localhost:5000/get_comments?report_id=${reportId}`);
      setComments((prev) => ({ ...prev, [reportId]: commentsResponse.data }));
      setNewComment((prev) => ({ ...prev, [reportId]: '' }));
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to post comment');
    } finally {
      setCommentLoading(false);
    }
  };

  useEffect(() => {
    fetchReportsAndComments();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#e0f2fe] to-[#bfdbfe] p-4">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-3xl font-bold text-[#1e3a8a] mb-6 text-center">Community Reports</h2>
        {error && (
          <div className="mb-6 p-4 bg-red-100 text-red-700 rounded-lg text-center">
            <p>{error}</p>
          </div>
        )}
        {reports.length === 0 && !error ? (
          <p className="text-center text-[#1e3a8a] text-lg">No unresolved reports available at the moment.</p>
        ) : (
          <div className="grid gap-6 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
            {reports.map((report) => (
              <div key={report._id} className="bg-white p-6 rounded-lg shadow-lg hover:shadow-xl transition-all duration-300">
                <img
                  src={report.image}
                  alt="Water Issue"
                  className="w-full h-48 object-cover rounded-lg mb-4"
                  onError={(e) => (e.target.src = 'https://via.placeholder.com/150?text=Image+Not+Found')}
                />
                <p className="text-[#1e3a8a]"><strong>Status:</strong> {report.status}</p>
                <p className="text-[#1e3a8a]"><strong>Address:</strong> {report.address}</p>
                <p className="text-[#1e3a8a]"><strong>Confidence:</strong> {report.confidence}</p>
                <p className="text-[#1e3a8a]"><strong>Upvotes:</strong> {report.upvotes}</p>
                {isLoggedIn && token ? (
                  <button
                    onClick={() => handleUpvote(report._id)}
                    className="mt-4 w-full bg-[#2563eb] text-[#f0f9ff] p-2 rounded-lg font-semibold transition-all duration-300 hover:bg-[#1e40af] hover:shadow-lg"
                  >
                    Upvote
                  </button>
                ) : (
                  <Link to="/login">
                    <button className="mt-4 w-full bg-gray-300 text-gray-700 p-2 rounded-lg font-semibold cursor-pointer">
                      Log in to Upvote
                    </button>
                  </Link>
                )}
                <div className="mt-6">
                  <h4 className="text-lg font-semibold text-[#1e3a8a] mb-2">Comments</h4>
                  {comments[report._id]?.length > 0 ? (
                    comments[report._id].map((comment) => (
                      <div key={comment._id} className="bg-[#e0f2fe] p-3 rounded-lg mb-2 shadow">
                        <p className="text-[#1e3a8a] font-medium">{comment.user_name}</p>
                        <p className="text-[#1e3a8a]">{comment.comment}</p>
                        <p className="text-[#6b7280] text-sm">
                          {new Date(comment.created_at).toLocaleString()}
                        </p>
                      </div>
                    ))
                  ) : (
                    <p className="text-[#1e3a8a]">No comments yet.</p>
                  )}
                  {isLoggedIn && token ? (
                    <div className="mt-4">
                      <textarea
                        value={newComment[report._id] || ''}
                        onChange={(e) =>
                          setNewComment((prev) => ({ ...prev, [report._id]: e.target.value }))
                        }
                        placeholder="Add a comment..."
                        className="w-full p-3 border border-[#93c5fd] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#60a5fa] transition-all duration-300 hover:border-[#60a5fa]"
                        rows={3}
                      />
                      <button
                        onClick={() => handleCommentSubmit(report._id)}
                        disabled={commentLoading}
                        className={`mt-2 w-full bg-[#2563eb] text-[#f0f9ff] p-2 rounded-lg font-semibold transition-all duration-300 ${
                          commentLoading ? 'opacity-50 cursor-not-allowed' : 'hover:bg-[#1e40af] hover:shadow-lg'
                        }`}
                      >
                        {commentLoading ? 'Posting...' : 'Post Comment'}
                      </button>
                    </div>
                  ) : (
                    <Link to="/login">
                      <button className="mt-4 w-full bg-gray-300 text-gray-700 p-2 rounded-lg font-semibold cursor-pointer">
                        Log in to Comment
                      </button>
                    </Link>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
        <div className="mt-12">
          <CommunityLeaderboard />
        </div>
      </div>
    </div>
  );
};

export default CommunityDashboard;