<<<<<<< HEAD
import React from 'react';
import { Container, Navbar, Nav, InputGroup, FormControl, Button, Card, Pagination, Badge, Row, Col } from 'react-bootstrap';
import { BsSearch } from 'react-icons/bs';
import { useNavigate, Link } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import { useLocation } from 'react-router-dom';

function ResultsPage() {
    const routeLocation = useLocation();
    const searchResults = routeLocation.state.searchResults;

    let navigate = useNavigate();
    const handleSearch = (searchTerm) => {
        navigate(`/ResultsPage?query=${searchTerm}`);
    };

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

            <Container>
                {/* Search Bar */}
                <InputGroup className="mb-4 mt-3">
                    <FormControl
                        placeholder="Search"
                        aria-label="Search"
                        aria-describedby="basic-addon2"
                    />
                    <Button variant="outline-secondary" id="button-addon2" onClick={() => handleSearch()}>
                        <BsSearch />
                    </Button>
                </InputGroup>

                {/* Search Results */}
                <h2>Search Results</h2>
            <ul>
                {searchResults.map((result, index) => (
                    <li key={index}>{result}</li>
                ))}
            </ul>

                {/* Pagination */}
                <Container className="d-flex justify-content-center mt-4">
                    <Pagination>
                        <Pagination.Item>{1}</Pagination.Item>
                        <Pagination.Item>{2}</Pagination.Item>
                        {/* Add more items as needed */}
                    </Pagination>
                </Container>
            </Container>
        </>
    );
}

export default ResultsPage;
=======
import React from 'react';
import { Container, Navbar, Nav, InputGroup, FormControl, Button, Card, Pagination, Badge, Row, Col } from 'react-bootstrap';
import { BsSearch } from 'react-icons/bs';
import { useNavigate, Link } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import { useLocation } from 'react-router-dom';

function ResultsPage() {
    const routeLocation = useLocation();
    const searchResults = routeLocation.state.searchResults;

    let navigate = useNavigate();
    const handleSearch = (searchTerm) => {
        navigate(`/ResultsPage?query=${searchTerm}`);
    };

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

            <Container>
                {/* Search Bar */}
                <InputGroup className="mb-4 mt-3">
                    <FormControl
                        placeholder="Search"
                        aria-label="Search"
                        aria-describedby="basic-addon2"
                    />
                    <Button variant="outline-secondary" id="button-addon2" onClick={() => handleSearch()}>
                        <BsSearch />
                    </Button>
                </InputGroup>

                {/* Search Results */}
                <h2>Search Results</h2>
            <ul>
                {searchResults.map((result, index) => (
                    <li key={index}>{result}</li>
                ))}
            </ul>

                {/* Pagination */}
                <Container className="d-flex justify-content-center mt-4">
                    <Pagination>
                        <Pagination.Item>{1}</Pagination.Item>
                        <Pagination.Item>{2}</Pagination.Item>
                        {/* Add more items as needed */}
                    </Pagination>
                </Container>
            </Container>
        </>
    );
}

export default ResultsPage;
>>>>>>> 6133087ecf6901100dff6b4e943054074084bcd7
