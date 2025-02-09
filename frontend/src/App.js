// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './Home';
import Chat from './Chat';
import Matches from './Matches';
import MatchesTest from './MatchTest'; // Import the test page

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/matches" element={<Matches />} />
        <Route path="/matches-test" element={<MatchesTest />} /> {/* Test page */}
      </Routes>
    </Router>
  );
}

export default App;
