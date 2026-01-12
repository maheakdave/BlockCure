import React, {useContext} from 'react';
import { Link } from 'react-router-dom';
import './home.css'
import { AuthContext } from '../../Hooks/AuthProvider.tsx';
import logo from '../../BlockCure_LOGO.jpg';

const Homepage:React.FC = () => {
    const {loggedin,logout} = useContext(AuthContext);
    return (
            <div>
                <nav>
                  <div className='navbar'>  
                {
                     
                    loggedin?
                    
                        <ul> 
                            <li><Link to = "/upload"><button>Upload</button></Link></li>
                            <li><Link to = "/browse"><button>Browse</button></Link></li>
                            <li><button onClick={logout}>Logout</button></li>
                        </ul>
                    :
                    <ul>
                        <li><Link to = "/login"><button>Login</button></Link></li>
                    </ul>
                }
                </div>
                <img src = {logo} alt = "logo"/>
                <h1>Welcome to BlockCure</h1>
                </nav>
            </div>

    )
}

export default Homepage;