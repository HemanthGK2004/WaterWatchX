import React, { useState, useEffect } from "react";
import axios from "axios";

const BlockchainViewer = () => {
  const [chain, setChain] = useState([]);
  const [isValid, setIsValid] = useState(null);

  useEffect(() => {
    // Fetch the blockchain
    axios.get("http://localhost:5000/blockchain/chain")
      .then(response => {
        setChain(response.data.chain);
      })
      .catch(error => {
        console.error("Error fetching blockchain:", error);
      });

    // Validate the blockchain
    axios.get("http://localhost:5000/blockchain/validate")
      .then(response => {
        setIsValid(response.data.valid);
      })
      .catch(error => {
        console.error("Error validating blockchain:", error);
      });
  }, []);

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Blockchain Viewer</h1>
      <p className="mb-4">
        Blockchain Validity: {isValid === null ? "Checking..." : isValid ? "Valid" : "Invalid"}
      </p>
      <h2 className="text-xl font-semibold mb-2">Chain</h2>
      {chain.map(block => (
        <div key={block.index} className="border p-4 mb-4 rounded">
          <p><strong>Index:</strong> {block.index}</p>
          <p><strong>Timestamp:</strong> {new Date(block.timestamp * 1000).toLocaleString()}</p>
          <p><strong>Proof:</strong> {block.proof}</p>
          <p><strong>Previous Hash:</strong> {block.previous_hash}</p>
          <p><strong>Hash:</strong> {block.hash}</p>
          <h3 className="text-lg font-medium mt-2">Transactions</h3>
          {block.transactions.length > 0 ? (
            block.transactions.map((tx, idx) => (
              <div key={idx} className="ml-4">
                <p><strong>Report ID:</strong> {tx.report_id}</p>
                <p><strong>User Phone:</strong> {tx.user_phone}</p>
                <p><strong>Status:</strong> {tx.status}</p>
                {tx.resolved_image && <p><strong>Resolved Image:</strong> <a href={tx.resolved_image} target="_blank">View</a></p>}
              </div>
            ))
          ) : (
            <p>No transactions</p>
          )}
        </div>
      ))}
    </div>
  );
};

export default BlockchainViewer;