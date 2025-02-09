// src/LoadingGraph.js
import React from 'react';
import './LoadingGraph.css';

const LoadingGraph = () => {
  // Create an array of nodes to render.
  const nodes = [0, 1, 2, 3, 4];
  return (
    <div className="loading-graph-container">
      <div className="node-list">
        {nodes.map((_, index) => (
          <React.Fragment key={index}>
            <div className="node" style={{ animationDelay: `${index * 0.2}s` }}></div>
            {index < nodes.length - 1 && (
              <div className="connector" style={{ animationDelay: `${index * 0.2}s` }}></div>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

export default LoadingGraph;
