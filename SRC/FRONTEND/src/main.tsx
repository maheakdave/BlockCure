import React from 'react'
import ReactDOM from 'react-dom/client'
import Homepage from './components/HomePage/Home.tsx'
import {RouterProvider, createBrowserRouter} from 'react-router-dom'
import LoginForm from './components/StartForm/Login.tsx';
import ErrorElement from './components/Error/ErrorHandler.tsx';
import UploadForm from './components/UploadRecords/Upload.tsx';
import { AuthProvider } from './Hooks/AuthProvider.tsx';
import { BrowseRecords } from './components/Browse/Browser.tsx';
  
const router = createBrowserRouter([
  {
    path: "/",
    element: <Homepage />,
    errorElement: <ErrorElement/>,
  },
  {
    path:"/login",
    element:<LoginForm/>,
  },
  {
    path:"/upload",
    element:<UploadForm/>,
  },
  {
    path:"/browse",
    element:<BrowseRecords/>,
  }
]);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AuthProvider>
      <RouterProvider router = {router}/>
    </AuthProvider>
  </React.StrictMode>
)
