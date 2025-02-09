import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import "./Matches.css";

const dummyMatches = [
  {
    id: 1,
    title: "Nonprofit A",
    summary:
      "Nonprofit A helps underserved communities by providing educational programs and financial assistance.",
    url: "https://nonprofita.org",
  },
  {
    id: 2,
    title: "Nonprofit B",
    summary:
      "Nonprofit B focuses on healthcare initiatives for low-income families through community clinics and outreach programs.",
    url: "https://nonprofitb.org",
  },
  {
    id: 3,
    title: "Nonprofit C",
    summary:
      "Nonprofit C provides career training, mentorship, and educational workshops to empower youth.",
    url: "https://nonprofitc.org",
  },
];

// The only addition here is y: -150 in exitLike and exitDislike so that it "floats" upward.
const cardVariants = {
  initial: { opacity: 0, scale: 0.8, y: 50 },
  animate: { opacity: 1, scale: 1, y: 0 },
  exitLike: { opacity: 0, scale: 1.2, x: 500, y: -150, rotate: 15 },
  exitDislike: { opacity: 0, scale: 1.2, x: -500, y: -150, rotate: -15 },
};

const MatchesTest = () => {
  const [matches, setMatches] = useState(dummyMatches);
  const [animating, setAnimating] = useState(false);
  const [exitDirection, setExitDirection] = useState(null);

  // When Like is clicked, animate the top card off to the right & up.
  const handleLike = () => {
    if (animating) return;
    setAnimating(true);
    setExitDirection("exitLike");
  };

  // When Dislike is clicked, animate the top card off to the left & up.
  const handleDislike = () => {
    if (animating) return;
    setAnimating(true);
    setExitDirection("exitDislike");
  };

  // When a drag ends on the top card, remove it if dragged far enough.
  const handleDragEnd = (e, info, id) => {
    if (Math.abs(info.point.x) > 100) {
      const direction = info.point.x > 0 ? "exitLike" : "exitDislike";
      setAnimating(true);
      setExitDirection(direction);
    }
  };

  // Once the exit animation completes on the top card, remove it.
  const onAnimationComplete = () => {
    setMatches((prev) => prev.slice(0, prev.length - 1));
    setAnimating(false);
    setExitDirection(null);
  };

  return (
    <div className="matches-wrapper">
      <button
        className="dislike-button shimmer-button"
        onClick={handleDislike}
        disabled={animating}
      >
        <span>Dislike</span>
      </button>

      <div className="matches-container">
        <AnimatePresence onExitComplete={onAnimationComplete}>
          {matches.map((match, index) => {
            // The top (front) card is the last in the array.
            const isTop = index === matches.length - 1;
            // Compute a stacking offset for the cards behind the top one.
            const offset = matches.length - 1 - index;
            const stackedStyle = {
              scale: 1 - offset * 0.02,
              y: offset * 10,
              rotate: offset * (offset % 2 === 0 ? 2 : -2),
            };

            return (
              <motion.div
                key={match.id}
                className="match-card"
                style={{ position: "absolute" }}
                drag={isTop ? "x" : false}
                dragConstraints={{ left: 0, right: 0 }}
                onDragEnd={isTop ? (e, info) => handleDragEnd(e, info, match.id) : undefined}
                variants={isTop ? cardVariants : {}}
                initial={isTop ? "initial" : stackedStyle}
                animate={isTop ? "animate" : stackedStyle}
                exit={
                  isTop
                    ? exitDirection === "exitLike"
                      ? cardVariants.exitLike
                      : cardVariants.exitDislike
                    : {}
                }
                transition={{ duration: 0.5 }}
              >
                <h3>{match.title}</h3>
                <p>{match.summary}</p>
                <a
                  href={match.url}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Visit Website
                </a>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>

      <button
        className="like-button shimmer-button"
        onClick={handleLike}
        disabled={animating}
      >
        <span>Like</span>
      </button>

      {matches.length === 0 && (
        <p style={{ textAlign: "center", marginTop: "20px" }}>
          No more matches available.
        </p>
      )}
    </div>
  );
};

export default MatchesTest;
