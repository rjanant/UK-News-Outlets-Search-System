// QueryExpansion.js
import React, { useEffect, useState } from 'react';
import { fetchQueryExpansions, fetchQuerySuggestion } from './api'; // Adjust the import path as necessary

const QueryExpansion = ({ onQuerySelect }) => {
  const [expansions, setExpansions] = useState([]);

// // main function -> uncomment this to use the actual expansion
//   useEffect(() => {
//     const getExpansions = async () => {
//       const data = await fetchQuerySuggestion(searchQuery.trim());
//       setExpansions(data);
//     };
//     getExpansions();
//   }, []);

//test 
const replacementStrings = [
    "Trump",
    "US President",
    "Politics"
  ];
  useEffect(() => {
    const getExpansions = async () => {
      await fetchQueryExpansions(); // Fetch expansions, but we'll not use the response directly
      // Set expansions to the predefined replacement strings
      setExpansions(replacementStrings.map(query => ({ query })));
    };
    getExpansions();
  }, []);

  const expansionStyle = {
    display: 'flex',
    justifyContent: 'center',
    marginTop: '20px',
  };

  const buttonStyle = {
    backgroundColor: '#f0f0f0', // Light grey background
    color: '#333', // Dark grey text
    border: 'none',
    borderRadius: '5px',
    padding: '10px 20px',
    margin: '5px',
    cursor: 'pointer',
    fontSize: '16px',
    transition: 'background-color 0.3s',
  };

  return (
    <div style={expansionStyle}>
      <div>
        <span>Did you mean: </span>
        {expansions.map((expansion, index) => (
          <button
            key={index}
            style={buttonStyle}
            onClick={() => onQuerySelect(expansion.query)}
            onMouseOver={(e) => (e.target.style.backgroundColor = '#e0e0e0')} // Slightly darker grey on hover
            onMouseOut={(e) => (e.target.style.backgroundColor = '#f0f0f0')} // Revert on mouse out
          >
            {expansion.query}
          </button>
        ))}
      </div>
    </div>
  );
};

export default QueryExpansion;
