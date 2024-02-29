import React from 'react';
import ResultsPage from './ResultsPage';
import HowItWorks from './HowItWorks';
import logoImage from './logo.png'; // Ensure this is the correct path
import { Container, InputGroup, FormControl, Button, Navbar, Nav } from 'react-bootstrap';
import { BsSearch } from 'react-icons/bs';
import { useNavigate, BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
    let navigate = useNavigate();

    const handleSearchClick = () => {
        navigate('/ResultsPage');
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


            <Container className="d-flex flex-column justify-content-center align-items-center" style={{ minHeight: '80vh', backgroundColor: '#ffffff', paddingTop: '5vh' }}>
                <div className="text-center" style={{ position: 'relative', top: '-15vh' }}>
                    <img 
                        src={logoImage} 
                        alt="FactChecker Logo"
                        className="mb-2"
                        style={{ maxWidth: '350px', width: '100%' }}
                    />
                    <InputGroup className="mb-3" style={{ maxWidth: '2000px', width: '100%' }}>
                        <FormControl
                          placeholder="Search"
                          aria-label="Search"
                          aria-describedby="basic-addon2"
                          style={{ borderRadius: '20px 0 0 20px', height: '50px' }}
                        />
                        <Button 
                          variant="outline-secondary" 
                          id="button-addon2" 
                          style={{ borderRadius: '0 20px 20px 0', height: '50px' }}
                          onClick={handleSearchClick}
                        >
                            <BsSearch />
                        </Button>
                    </InputGroup>
                </div>
            </Container>

            <footer className="text-center bg-light py-3">
                <Container>
                    © {new Date().getFullYear()} FactChecker - All Rights Reserved
                    <div>
                        <a href="#privacy-policy">Privacy Policy</a> | <a href="#terms-of-service">Terms of Service</a>
                    </div>
                    <div>
                        Follow us: 
                        <a href="#twitter" className="ms-2">Twitter</a> | 
                        <a href="#facebook" className="ms-2">Facebook</a> | 
                        <a href="#instagram" className="ms-2">Instagram</a>
                    </div>

                    {/* Delete this after testing */}
                    <button
                    onClick={async () => {
                        const result = await fetch(
                        `${process.env.REACT_APP_ENDPOINT_URL}/search/test`,
                        {
                            method: "POST",
                            headers: {
                            "Content-Type": "application/json",
                            },
                            body: JSON.stringify({
                            field: "test",
                            }),
                        }
                        );
                        const data = await result.json();
                        console.log(data);
                    }}
                    >
                        Test
                    </button>
                    
                </Container>
            </footer>
        </>
    );
}

function AppWrapper() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<App />} />
                <Route path="/ResultsPage" element={<ResultsPage />} />
                <Route path="/how-it-works" element={<HowItWorks />} />
            </Routes>
        </BrowserRouter>
    );
}

export default AppWrapper;
