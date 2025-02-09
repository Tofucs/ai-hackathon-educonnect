import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './Matches.css';

function Matches() {
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(true);
  const [animating, setAnimating] = useState(false);
  const [exitDirection, setExitDirection] = useState(null);

  // Fetch matches from the backend when the component mounts.
  useEffect(() => {
    const searchQuery = localStorage.getItem('searchQuery') || 'nonprofit';
    const fetchMatches = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/api/matches', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ search_query: searchQuery })
        });
        if (!response.ok) {
          console.error("Matches API error:", response.statusText);
          setLoading(false);
          return;
        }
        const data = await response.json();
        setMatches(data.matches);
      } catch (error) {
        console.error("Error fetching matches:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchMatches();
  }, []);

  // Framer Motion variants for card animations.
  const cardVariants = {
    initial: { opacity: 0, scale: 0.8, y: 50 },
    animate: { opacity: 1, scale: 1, y: 0 },
    exitLike: { opacity: 0, scale: 1.2, x: 500, rotate: 15 },
    exitDislike: { opacity: 0, scale: 1.2, x: -500, rotate: -15 },
  };

  // Handlers for Like and Dislike actions.
  const handleLike = () => {
    if (animating) return;
    setAnimating(true);
    setExitDirection('exitLike');
  };

  const handleDislike = () => {
    if (animating) return;
    setAnimating(true);
    setExitDirection('exitDislike');
  };

  // Once the exit animation completes, remove the current match.
  const onAnimationComplete = () => {
    setMatches(prev => prev.slice(1));
    setAnimating(false);
    setExitDirection(null);
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <div className="loading-text">Loading matches...</div>
      </div>
    );
  }

  // Show only one nonprofit at a time.
  const currentMatch = matches[0];

  return (
    <div className="matches-container">
      <AnimatePresence onExitComplete={onAnimationComplete}>
        {currentMatch && (
          <motion.div
            key={currentMatch.title}  // use a unique key if available
            className="match-card"
            variants={cardVariants}
            initial="initial"
            animate="animate"
            exit={exitDirection}
            transition={{ duration: 0.5 }}
          >
            <h3>{currentMatch.title}</h3>
            <p>{currentMatch.summary}</p>
            <a href={currentMatch.url} target="_blank" rel="noopener noreferrer">
              Visit Website
            </a>
          </motion.div>
        )}
      </AnimatePresence>
      <div className="button-group">
        <button className="dislike-button" onClick={handleDislike} disabled={animating}>
          Dislike
        </button>
        <button className="like-button" onClick={handleLike} disabled={animating}>
          Like
        </button>
      </div>
    </div>
  );
}

export default Matches;
