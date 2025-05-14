import axios from 'axios';
import React, { useEffect, useState } from 'react';

const CommunityLeaderboard = () => {
  const [leaderboard, setLeaderboard] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchLeaderboard = async () => {
      try {
        const response = await axios.get('http://localhost:5000/community_leaderboard');
        setLeaderboard(response.data);
      } catch (err) {
        setError(err.response?.data?.error || 'An error occurred while fetching leaderboard');
      }
    };
    fetchLeaderboard();
  }, []);

  return (
    <div className="py-12 px-6 bg-gradient-to-b from-[#e0f2fe] to-[#bfdbfe]">
      <div className="max-w-7xl mx-auto">
        <h2 className="text-3xl font-extrabold text-[#1e3a8a] mb-8 text-center tracking-tight">
          Community Leaderboard
        </h2>
        {error && (
          <div className="mb-6 p-4 bg-red-100 text-red-700 rounded-lg">
            <p>Error: {error}</p>
          </div>
        )}
        {leaderboard.length > 0 ? (
          <div className="bg-[#e0f2fe] p-6 rounded-xl shadow-2xl">
            <table className="w-full text-left">
              <thead>
                <tr className="text-[#1e3a8a] border-b border-[#93c5fd]">
                  <th className="p-4 font-semibold">Rank</th>
                  <th className="p-4 font-semibold">User</th>
                  <th className="p-4 font-semibold">Total Upvotes</th>
                  <th className="p-4 font-semibold">Reports Submitted</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((entry, index) => (
                  <tr
                    key={entry.user_phone}
                    className="border-b border-[#93c5fd] hover:bg-[#bfdbfe] transition-all duration-200"
                  >
                    <td className="p-4">{index + 1}</td>
                    <td className="p-4">{entry.user_name}</td>
                    <td className="p-4">{entry.total_upvotes}</td>
                    <td className="p-4">{entry.report_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-[#1e3a8a] text-center">No leaderboard data available.</p>
        )}
      </div>
    </div>
  );
};

export default CommunityLeaderboard;