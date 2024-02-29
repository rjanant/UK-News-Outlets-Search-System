import React from 'react';
import {
  Container, Navbar, Nav, InputGroup,
  FormControl, Button, Card, Pagination, Badge, Row, Col
} from 'react-bootstrap';
import { BsSearch } from 'react-icons/bs';
import { useNavigate, Link } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';

function ResultsPage() {
    let navigate = useNavigate();

    const handleSearch = (searchTerm) => {
        navigate(`/ResultsPage?query=${searchTerm}`);
    };

    // Placeholder data
    const results = [
        { title: "Result 1", description: "Description of result 1", label: "Label 1" },
        { title: "Result 2", description: "Description of result 2", label: "Label 2" },
        { title: "Result 3", description: "Description of result 3", label: "Label 3" }
    ];

    const similarWords = ["example", "sample", "demo"]; // Placeholder similar words

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

                 {/* Similar Words Placeholder */}
                 <Row className="mb-4 gx-2">
                    {similarWords.map((word, index) => (
                        <Col key={index} xs="auto">
                            <Button variant="outline-secondary">{word}</Button>
                        </Col>
                    ))}
                </Row>

                {/* Results List */}
                {results.map((result, index) => (
                    <Card key={index} className="mb-3">
                        <Card.Body>
                            <Card.Title>{result.title}</Card.Title>
                            <Card.Text>{result.description}</Card.Text>
                            <Badge bg="secondary">{result.label}</Badge>
                        </Card.Body>
                    </Card>
                ))}


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
