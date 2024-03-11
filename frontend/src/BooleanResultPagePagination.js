import React, { useState, useEffect } from 'react';
import { fetchSearchBoolean } from './api';
import { Container, Navbar, Nav, InputGroup, FormControl, Button, Card, Spinner, Badge, Form, Row, Col, Pagination } from 'react-bootstrap';
import { BsSearch } from 'react-icons/bs';
import { useNavigate, Link, useLocation } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import QueryExpansion from './queryExpansion';


const SentimentBadge = ({ sentiments }) => {
  const getColorWithIntensity = (baseColor, value) => {
    const intensityFactor = 0.5 + (value * 0.5); 
    return `rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, ${intensityFactor})`;
  };

  const getFontColor = (backgroundColor) => {
    const brightness = (backgroundColor.r * 299 + backgroundColor.g * 587 + backgroundColor.b * 114) / 1000;
    return brightness > 125 ? 'black' : 'white';
  };

  const totalValue = sentiments.reduce((sum, { value }) => sum + value, 0);

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      height: '35px',
      borderRadius: '20px',
      overflow: 'hidden',
      maxWidth: '60%',
      marginLeft: '0',
      alignSelf: 'flex-start',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
    }}>
      {sentiments.map(({ type, value }, index) => {
        if (value === 0) return null;
        const baseColors = {
          positive: { r: 0, g: 128, b: 0 },
          neutral: { r: 128, g: 128, b: 128 },
          negative: { r: 255, g: 0, b: 0 },
        };
        const widthPercent = (value / totalValue) * 100;
        const backgroundColor = getColorWithIntensity(baseColors[type], value);
        const fontColor = getFontColor(backgroundColor);
        
        const tooltipText = `${type.charAt(0).toUpperCase() + type.slice(1)}: ${(value * 100).toFixed(1)}%`;

        return (
          <div key={index} style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            width: `${widthPercent}%`,
            backgroundColor: backgroundColor,
            color: fontColor,
            fontSize: '1rem',
            transition: 'width 0.3s ease',
          }} title={tooltipText}>
            <span style={{ padding: '0 5px' }}>{(value * 100).toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
};


function BooleanResultsPage() {

    // const [darkMode, setDarkMode] = useState(false);
    // useEffect(() => {
    //   // Check for saved user preference, if available
    //   const isDarkMode = localStorage.getItem('darkMode') === 'true';
    //   setDarkMode(isDarkMode);
    // }, []);
  
    // const toggleDarkMode = () => {
    //   setDarkMode(!darkMode);
    //   localStorage.setItem('darkMode', !darkMode); // Save preference
    // };
  
    const { searchResults } = useLocation().state || { searchResults: [] };
    let navigate = useNavigate();
    const [searchQuery, setSearchQuery] = useState('');
    const [loading, setLoading] = useState(false);
    const [filterYear, setFilterYear] = useState('all');
    const [sentimentFilter, setSentimentFilter] = useState('all');
    const [sourceFilter, setSourceFilter] = useState('all');
    const [currentPage, setCurrentPage] = useState(1);
    const [resultsPerPage, setResultsPerPage] = useState(5);


    useEffect(() => {
      const searchParams = new URLSearchParams(window.location.search);
      const query = searchParams.get('q');
      if (query) {
        setSearchQuery(query);
        performSearch(query);
      }
    }, []);
  
    const performSearch = async (searchTerm) => {
      setLoading(true);
      try {
        const results = await fetchSearchBoolean(searchTerm);
        navigate('/BooleanResultsPage', { state: { searchResults: results, searchType: 'boolean' } });
      } catch (error) {
        console.error('Error fetching boolean search results:', error);
      }
      setLoading(false);
    };
  
    const handleSearch = (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        performSearch(searchQuery);
      }
    };
  
    const handleSearchClick = () => {
      performSearch(searchQuery);
    };
  
    const handleQueryExpansionSelect = (newQuery) => {
      setSearchQuery(newQuery);
      performSearch(newQuery);
    };

    
    const currentYear = new Date().getFullYear();
    const years = Array.from({ length: currentYear - 2020 + 1 }, (_, i) => String(currentYear - i));
    
    const uniqueSources = Array.from(new Set(searchResults.map(result => result.source)));
    const filteredResults = searchResults
          .filter(result => {
          const resultYear = new Date(result.date).getFullYear().toString();
          const meetsYearCriteria = filterYear === 'all' || resultYear === filterYear;
          const meetsSourceCriteria = sourceFilter === 'all' || result.source === sourceFilter;
    if (!meetsYearCriteria || !meetsSourceCriteria) return false;
    return true;
  })
  .sort((a, b) => {
    const yearA = new Date(a.date).getFullYear();
    const yearB = new Date(b.date).getFullYear();
    if (yearA !== yearB) {
      return yearB - yearA; 
    }

    // Parse sentiment data before using it as it is in different format now with an /
    const sentimentA = JSON.parse(a.sentiment);
    const sentimentB = JSON.parse(b.sentiment);
    
    const sentimentScoreA = sentimentA.find(item => item.includes(sentimentFilter))?.split(': ')[1] || '0';
    const sentimentScoreB = sentimentB.find(item => item.includes(sentimentFilter))?.split(': ')[1] || '0';

    return parseFloat(sentimentScoreB) - parseFloat(sentimentScoreA);
  });


    const indexOfLastResult = currentPage * resultsPerPage;
    const indexOfFirstResult = indexOfLastResult - resultsPerPage;
    const currentResults = filteredResults.slice(indexOfFirstResult, indexOfLastResult);

    const ColorCodingGuide = () => (
        <div style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
          <Badge bg="success" text="dark">Positive</Badge>
          <Badge bg="secondary" text="dark">Neutral</Badge>
          <Badge bg="danger" text="dark">Negative</Badge>
        </div>
      );
    
      return (
        
        <>
        
          <Navbar bg="light" expand="lg">
            <Container>
              <Navbar.Brand as={Link} to="/">FactChecker</Navbar.Brand>
              <Navbar.Toggle aria-controls="basic-navbar-nav" />
              <Navbar.Collapse id="basic-navbar-nav">
                <Nav className="me-auto">
                  <Nav.Link as={Link} to="/">Home</Nav.Link>
                  <Nav.Link as={Link} to="/how-it-works">How It Works</Nav.Link>
                </Nav>
              </Navbar.Collapse>
            </Container>
          </Navbar>
      
          <QueryExpansion onQuerySelect={handleQueryExpansionSelect} currentQuery={searchQuery} />
      
          <Container>
            <Row>
              <Col md={3} className="filter-sidebar">
                <h4>Filter Results</h4>
                <Form>
                <Form.Group controlId="filterYear">
                  <Form.Label>Year</Form.Label>
                  <Form.Select value={filterYear} onChange={e => setFilterYear(e.target.value)}>
                    <option value="all">All Years</option>
                    {years.map(year => (
                      <option key={year} value={year.toString()}>{year}</option>
                    ))}
                  </Form.Select>
                </Form.Group>
                <Form.Group controlId="sentimentFilter">
                  <Form.Label>Sentiment</Form.Label>
                  <Form.Select value={sentimentFilter} onChange={e => setSentimentFilter(e.target.value)}>
                    <option value="all">All Sentiments</option>
                    <option value="positive">Positive</option>
                    <option value="neutral">Neutral</option>
                    <option value="negative">Negative</option>
                  </Form.Select>
                </Form.Group>
                <Form.Group controlId="sourceFilter">
                    <Form.Label>Source</Form.Label>
                    <Form.Select value={sourceFilter} onChange={e => setSourceFilter(e.target.value)}>
                        <option value="all">All Sources</option>
                        {uniqueSources.map((source, index) => (
                        <option key={index} value={source}>{source}</option>
                            ))}
                        </Form.Select>
                </Form.Group>
                </Form>
              </Col>
              <Col md={9}>
                <h2>Boolean Search Results</h2>
                <ColorCodingGuide /> 
                {loading ? (
                  <Spinner animation="border" role="status">
                    <span className="visually-hidden">Loading...</span>
                  </Spinner>
                ) : (
                  currentResults.length > 0 ? currentResults.map((result, index) => (
                    <Card key={index} className="mb-3">
                      <Card.Body>
                        <Card.Title>{result.title}</Card.Title>
                        <SentimentBadge sentiments={JSON.parse(result.sentiment).map(item => {
                        const parts = item.split(':');
                        return { type: parts[0].trim(), value: parseFloat(parts[1]) };
                          })} />
                        <Card.Text>
                          <strong>Date:</strong> {result.date}<br />
                          <strong>Summary:</strong> {result.summary}
                        </Card.Text>
                        <Button variant="primary" href={result.url}>Read More</Button>
                      </Card.Body>
                    </Card>
                  )) : <div>No results found for the selected filters.</div>
                )}
                <div className="pagination-controls d-flex justify-content-center align-items-center mt-3">
                  {Array.from({ length: Math.ceil(filteredResults.length / resultsPerPage) }, (_, index) => (
                    <Button 
                        key={index} 
                        variant={currentPage === index + 1 ? "primary" : "light"} 
                        onClick={() => setCurrentPage(index + 1)}
                        className="me-2"
                    >
                    {index + 1}
                    </Button>
                    ))}
                </div>
            </Col>
            </Row>
          </Container>
        </>
      ); 
}
export default BooleanResultsPage;