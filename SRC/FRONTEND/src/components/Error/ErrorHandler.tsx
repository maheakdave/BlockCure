import { useRouteError } from "react-router-dom";
import './error.css';


const ErrorElement:React.FC = () =>{

    const error:any = useRouteError();
    return (
    <div>
      <h1>Oops!</h1>
      <p>Sorry, an unexpected error has occurred.</p>
      <p>
        <i>{error.statusText || error.message}</i>
      </p>
    </div>
    );
}

export default ErrorElement;