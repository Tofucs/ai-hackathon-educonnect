import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import './Home.css';

const SWAP_DELAY_IN_MS = 5000;
const LETTER_DELAY = 0.03;
const FACTS = [
  "Over 50% of non-profits focus on improving access to education.",
  "Educational non-profits often offer free mentorship programs.",
  "Some non-profits provide scholarships for underrepresented students.",
  "Teacher training programs are key offerings from many educational non-profits.",
  "Community-based learning programs are gaining popularity among non-profits."
];

function Home() {
  const navigate = useNavigate();
  const [factIndex, setFactIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setFactIndex((prevIndex) => (prevIndex + 1) % FACTS.length);
    }, SWAP_DELAY_IN_MS);
    return () => clearInterval(interval);
  }, []);

  const handleStartClick = () => {
    navigate('/chat');
  };

  return (
    <div className="home-container">
      {/* Logo Section */}
      <div className="logo-container">
        <img src="/EduConnect.png" alt="EduConnect" className="logo" />
      </div>

      {/* Animated Intro Section */}
      <motion.div
        className="intro-card"
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <p className="intro-tag">/ EduConnect.AI</p>
        <hr className="divider" />
        <p className="intro-text">
          <strong>Helping bring educational benefits to everyone.</strong>
        </p>
        <div className="animated-fact">
          <span className="fact-label">DID YOU KNOW:</span>{" "}
          {FACTS[factIndex].split("").map((char, i) => (
            <motion.span
              key={`${factIndex}-${i}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: i * LETTER_DELAY, duration: 0.1 }}
            >
              {char}
            </motion.span>
          ))}
        </div>
        <hr className="divider" />
      </motion.div>

      {/* Content Section */}
      <motion.div
        className="home-content"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3, duration: 0.5 }}
      >
        <div className="image-container">
          <img src="/nonprofit.jpg" alt="Nonprofit" className="content-image" />
        </div>
        <h2 className="subheading">What is EduConnect.AI?</h2>
        <p className="description">
          EduConnect.AI is an AI solution that helps connect underserviced students and school districts to educational non-profits to help bridge the education gap.
        </p>
        <h2 className="subheading">How do you use EduConnect.AI?</h2>
        <p className="description">
          EduConnect.AI begins by asking you a few initial questions to tailor its suggestions. You can rate the recommendations provided, and these ratings help refine our AI agent to deliver increasingly accurate suggestions, guiding you to your ideal non-profit. Once you find the perfect match, EduConnect will assist in drafting a message to help you connect with the organization. Ready to start? Click the button below!
        </p>
        <div className="button-container">
          <button className="start-button" onClick={handleStartClick}>
            Start Connecting!
          </button>
        </div>
      </motion.div>
    </div>
  );
}

export default Home;
